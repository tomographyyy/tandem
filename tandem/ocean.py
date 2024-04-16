"""
Copyright 2024 Tomohiro TAKAGAWA

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from petsc4py import PETSc
import time
import numpy as np
from tandem.solidearth import SolidEarth
from tandem.dmstag import DMStagDA, DMDAHierarchy, DMStagVariableSet, DMStagBase, DMDAStencil
from tandem.station import Station, Angle
from tandem.okada import Okada
from pyproj import Geod
import xarray as xr
import os
from mpi4py import MPI
from scipy import signal

class Ocean(object):
    def get_gravity(self, lats, isUniform=False):
        """
        get gravity acceleration values

        Args:
            lats (ndarray): latitudes (radian)
            isUniform (bool): uniform gravity or not

        Returns:
            gravities (ndarray): gravity acceleration values (m/s)
        """
        if not isUniform:
            a = 6378137 # equatorial semi-axes [m]
            f = 1.0 / 298.257222101 # flattening
            ge = 9.7803253359 # gravity at the equator [m/s2]
            gp = 9.8321849378 # gravity at the poles   [m/s2]
            b = a * (1 - f)
            e2 = 1 - (b/a)**2 
            k = (b * gp - a * ge) / (a * ge)
            sin2 = np.sin(lats)**2
            g = ge * (1 + k * sin2) / np.sqrt(1 - e2 * sin2)
        else:
            g = np.ones_like(lats) * 9.80665
        return g
    def get_radius_by_lat(self, lat):
        a = self.geod.a
        b = self.geod.b
        ac2 = (a * np.cos(lat))**2
        bs2 = (b * np.sin(lat))**2
        numerator = a**2 * ac2 + b**2 * bs2
        denominator = ac2 + bs2
        R = np.sqrt(numerator / denominator)
        return R
    def get_mean_radius(self, lat0, lat1, num=2**12):
        lats = np.linspace(lat0, lat1, num=num)
        Rs = self.get_radius_by_lat(lats)
        return np.mean(Rs)
    def get_linear_taper(self, n, width):
        taper = np.minimum(np.arange(n), n - 1 - np.arange(n))
        taper = np.minimum(taper / width, 1)
        return taper
    def set_taper(self, width):
        self.taper_xM = self.get_linear_taper(self.shape[0] + 1, width)
        self.taper_yN = self.get_linear_taper(self.shape[1] + 1, width)
        self.taper_xh = 0.5 * (self.taper_xM[1:] + self.taper_xM[:-1])
        self.taper_yh = 0.5 * (self.taper_yN[1:] + self.taper_yN[:-1])

    def __init__(self, shape, extent=(0,1,1,0), 
                CLV=0, damping_factor=0.1, hyperbolic_r=3, 
                has_Boussinesq=False, outpath="", filter_radius=5e3, 
                Manning=0., Nonlinear=False):
        self.comm = PETSc.COMM_WORLD
        self.rank = self.comm.rank
        self.geod = Geod(ellps="WGS84")
        self.R = self.geod.a
        self.damping_factor = damping_factor
        self.hyperbolic_r = hyperbolic_r
        self.has_Boussinesq = has_Boussinesq
        self.solid_earth = None
        self.outpath = outpath
        self.filter_radius = filter_radius
        self.shape = shape
        self.extent = extent
        self.CLV = CLV
        self.dx = np.abs(extent[1] - extent[0]) / shape[0]
        self.dy = np.abs(extent[3] - extent[2]) / shape[1]
        self.Manning=Manning
        self.Nonlinear=Nonlinear
        # DMDA global
        self.d = DMStagDA() # depth
        self.h = DMStagDA() # water surface height
        self.M = DMStagDA() # eastward flux
        self.N = DMStagDA() # southward flux
        self.z = DMStagDA() # bottom displacement
        self.Phai = DMStagDA() # variables for Poisson solver
        self.b = DMStagDA() # variables for Poisson solver
        self.b0 = DMStagDA()
        self.hmax = DMStagDA()
        if self.Nonlinear:
            self.tmpM0 = DMStagDA()
            self.tmpM1 = DMStagDA()
            self.tmpM2 = DMStagDA()
            self.tmpN0 = DMStagDA()
            self.tmpN1 = DMStagDA()
            self.tmpN2 = DMStagDA()
        
        # DMDA local
        isLocal=False
        self.h_pre = DMStagDA(local=isLocal)
        self.M_pre = DMStagDA(local=isLocal)
        self.N_pre = DMStagDA(local=isLocal)
        self.AMBN = DMStagDA(local=isLocal)
        self.comp = DMStagDA(local=isLocal)
        self.limit = DMStagDA(local=isLocal)
        self.h_star = DMStagDA(local=isLocal)
        self.z_pre = DMStagDA(local=isLocal)

        # DMDA temporal
        self.dM = DMStagDA(local=True)
        self.dN = DMStagDA(local=True)
        self.hx = DMStagDA(local=True)
        self.hy = DMStagDA(local=True)

        self.elements = (self.d, self.h, self.h_pre, 
                         self.z, self.z_pre, self.h_star, 
                         self.Phai, self.b, self.AMBN, 
                         self.comp, self.limit, self.b0,
                         self.hmax)
        self.left_and_downs = [ (self.M, self.N), 
                                (self.M_pre, self.N_pre),
                                (self.dM, self.dN),
                                (self.hx, self.hy)]
        if self.Nonlinear:
            self.left_and_downs.append((self.tmpM0, self.tmpN0))
            self.left_and_downs.append((self.tmpM1, self.tmpN1))
            self.left_and_downs.append((self.tmpM2, self.tmpN2))
        self.varSet = DMStagVariableSet()
        for element in self.elements:
            self.varSet.add_element(element)
        for left_and_down in self.left_and_downs:
            self.varSet.add_left_and_down(*left_and_down)
        dmstag = DMStagBase(self.varSet, self.shape, self.extent)
         
        for var in self.varSet.variables:
            var.setup(dmstag)
        
        if self.CLV > 0:
            self.dmHierarchy = DMDAHierarchy()
            self.dmHierarchy.setup(self.h.da.getSizes(),nlevel=self.CLV)
            self.h_star_coarse_pre = np.zeros_like(self.dmHierarchy.get_coarse(self.z))
            self.z_coarse_pre = np.zeros_like(self.dmHierarchy.get_coarse(self.z))

        self.xM = np.linspace(extent[0], extent[1], num=shape[0] + 1)
        self.yN = np.linspace(extent[2], extent[3], num=shape[1] + 1)
        self.xN = 0.5 * (self.xM[1:] + self.xM[:-1])
        self.yM = 0.5 * (self.yN[1:] + self.yN[:-1])
        self.gM = self.get_gravity(self.yM)
        self.gN = self.get_gravity(self.yN)

        rx = self.R * self.dx * np.min(np.cos(self.yM))
        ry = self.R * self.dy
        self.stencil_width = int(np.floor(filter_radius / np.minimum(rx, ry)))
        if self.rank==0:
            print("stencil_width", self.stencil_width)
        self.h_stencil = DMDAStencil() 
        self.h_stencil.setup(self.h, stencil_width=self.stencil_width)

    def load_bathymetry(self, filename, depth=False, lon="lon", lat="lat", z="z"):
        nc = xr.open_mfdataset(filename, combine='by_coords', join="override", data_vars="minimal")
        x_src = np.deg2rad(nc[lon].data)
        y_src = np.deg2rad(nc[lat].data)
        x_extent = self.extent[:2]
        y_extent = self.extent[2:]
        x_mid = self.xN[1]
        y_mid = self.yN[1]
        angle = Angle()
        (i0, i1), (j0, j1)  = self.d.da.getRanges()
        (iMax, jMax) = self.d.da.getSizes()
        x_dst = self.xN[i0:i1]
        y_dst = self.yM[j0:j1]
        i_near = angle.get_nearest_indexes(x_dst, x_src, x_extent, x_mid)
        j_near = angle.get_nearest_indexes(y_dst, y_src, y_extent, y_mid)
        #print("x_dst:", x_dst)
        #print("x_src:", x_src)
        #print("x_extent:", x_extent)
        #print("x_mid:", x_mid)
        #print("i:", i_near)
        #print("j:", j_near, flush=True)
        elev_local = nc[z][j_near, i_near]
        if depth:
            elev_local = nc[z][j_near, i_near]
        else:
            elev_local = nc[z][j_near, i_near] * -1
        self.d.vecArray[i0:i1,j0:j1] = np.maximum(elev_local.T, 0)
        if i0==0:
            self.d.vecArray[i0-1,j0:j1] = self.d.vecArray[i0,j0:j1]
        if i1==iMax:
            self.d.vecArray[i1,j0:j1] = self.d.vecArray[i1-1,j0:j1]
        if j0==0:
            self.d.vecArray[i0:i1,j0-1] = self.d.vecArray[i0:i1,j0]
        if j1==jMax:
            self.d.vecArray[i0:i1,j1] = self.d.vecArray[i0:i1,j1-1]
        self.d.local_to_local()

    def set_solid_earth(self, solid_earth):
        self.solid_earth = solid_earth

    def setup_station(self, csv_file, taper_width):
        (i0, i1), (j0, j1)  = self.h.da.getRanges()
        (iMax, jMax) = self.h.da.getSizes()
        i0 = np.maximum(i0, taper_width)
        j0 = np.maximum(j0, taper_width)
        i1 = np.minimum(i1, iMax - 1 - taper_width)
        j1 = np.minimum(j1, jMax - 1 - taper_width)
        xextent = (self.xM[i0], self.xM[i1+1])
        yextent = (self.yN[j0], self.yN[j1+1])
        xmid = self.xM[i0+1]
        ymid = self.yM[j0+1]
        self.station = Station(csv_file)
        self.station.set_nearest_index(
            self.xN, self.yM,
            xextent, yextent, xmid, ymid, (i0, i1), (j0, j1))
        self.station.save_csv(f"{self.outpath}/station{self.rank:04}.csv")
        #wx = wy = np.deg2rad(3.0)
        #self.is_near_source = self.is_near_source(self.xN, self.yM, xextent, yextent, wx=wx, wy=wy)

    def setup_ksp(self, is_adjoint=False):
        self.ksp = PETSc.KSP().create(comm = self.comm)
        self.ksp.setDM(self.Phai.da)
        t0 = time.time()
        if is_adjoint:
            #print("adjoint")
            self.ksp.setComputeOperators(self.set_Poisson_adj)
        else:
            #print("forward")
            self.ksp.setComputeOperators(self.set_Poisson)
        self.ksp.setType('cg')
        self.ksp.getPC().setType('hypre') 
        self.ksp.setTolerances(rtol=1e-15, atol=None, divtol=None, max_it=None)
        self.ksp.setFromOptions()
        self.ksp.setUp()
        self.comm.Barrier()
        if self.rank==0:
            duration = time.time() - t0
            print(f"setup ksp operator: {duration} sec")

    def get_all_variables(self):
        data=[]
        variables = [self.M_pre, self.N_pre, self.comp, 
                     self.limit, self.d, self.h, self.z, 
                     self.h_pre, self.z_pre, self.M, self.N, 
                     self.b, self.Phai, self.AMBN, self.h_star]
        for variable in variables:
            variable.gather()
            if self.rank==0:
                data.append(variable.rank0[:])
        return data

    def set_all_variables(self, data):
        variables = [self.M_pre, self.N_pre, self.comp, 
                     self.limit, self.d, self.h, self.z, 
                     self.h_pre, self.z_pre, self.M, self.N, 
                     self.b, self.Phai, self.AMBN, self.h_star]
        if self.rank==0:
            for i in range(len(variables)):
                variables[i].rank0[:] = data[i]
        for variable in variables:
            variable.scatter()
            variable.set_boundary(boundary_type="copy")

    def set_dMN(self):
        self.dM.set_meanx(self.d, boundary_type="none", positive_restriction=True)
        self.dN.set_meany(self.d, boundary_type="none", positive_restriction=True)

    def set_comp(self, compressible=True, vp=1450):
        (i0, i1), (j0, j1)  = self.comp.da.getRanges()
        (iMax, jMax) = self.comp.da.getSizes()
        self.comp.vecArray[i0:i1,j0:j1] = 1
        dep = np.maximum(self.d.vecArray[i0:i1,j0:j1], 0)
        grv = self.gM[j0:j1]
        if compressible:
            self.comp.vecArray[i0:i1,j0:j1] -= dep * grv / (2 * vp**2)
        self.comp.local_to_local()

    def set_d_margin(self, width):
        (i0, i1), (j0, j1)  = self.d.da.getRanges()
        (iMax, jMax) = self.d.da.getSizes()
        w = width
        if i0==0:
            for i in range(w):
                self.d.vecArray[i,j0:j1] = self.d.vecArray[w,j0:j1]
        if i1==iMax:
            for i in range(iMax-w, iMax):
                self.d.vecArray[i,j0:j1] = self.d.vecArray[iMax-1-w,j0:j1]
        if j0==0:
            for j in range(w):
                self.d.vecArray[i0:i1,j] = self.d.vecArray[i0:i1,w]
        if j1==jMax:
            for j in range(jMax-w, jMax):
                self.d.vecArray[i0:i1,j] = self.d.vecArray[i0:i1,jMax-1-w]
        self.d.local_to_local()

    def set_d_positive(self):
        (i0, i1), (j0, j1)  = self.d.da.getRanges()
        (iMax, jMax) = self.d.da.getSizes()
        self.d.vecArray[i0:i1,j0:j1] = np.maximum(0, self.d.vecArray[i0:i1,j0:j1])
        self.d.local_to_local()

    def setup(self, taper_width, station_csv=None, compressible=False):
        self.set_d_positive()
        self.set_d_margin(width=taper_width)
        self.set_dMN()
        self.set_comp(compressible=compressible)
        self.set_taper(width=taper_width)
        #self.setup_ksp()
        self.setup_station(station_csv, taper_width)

    def update_MN_by_h(self, dt=1):
        if self.Nonlinear:
            Dmin = 0.1
            Dmax = 500
            max_slope = 2
            
            # advection of M
            i, j = self.M.get_slice()
            taperI = self.taper_xM[i[0]].reshape(-1,1)
            taperJ = self.taper_yh[j[0]].reshape(1,-1)
            tapering_factor_M = taperI.dot(taperJ)
            
            DMinv = self.tmpM0
            _dw = self.d.vecArray[i[-1],j[0]]
            _de = self.d.vecArray[i[ 0],j[0]]
            _d = np.minimum(self.d.vecArray[i[-1],j[0]], self.d.vecArray[i[0],j[0]])
            _h = np.maximum((self.h.vecArray[i[-1],j[0]] + self.h_pre.vecArray[i[-1],j[0]]) / 2,
                            (self.h.vecArray[i[ 0],j[0]] + self.h_pre.vecArray[i[ 0],j[0]]) / 2)
            
            _D = _d + _h
            DMinv.vecArray[i[0], j[0]] = np.divide(1, _D, out=np.zeros_like(_D), where=_D>Dmin)
            DMinv.local_to_local()

            slope_w = np.divide(DMinv.vecArray[i[0], j[0]], DMinv.vecArray[i[-1], j[0]], 
                                out=np.full(_dw.shape, 100.0), 
                                where=(DMinv.vecArray[i[0], j[0]] * DMinv.vecArray[i[-1], j[0]] > 0))
            slope_e = np.divide(DMinv.vecArray[i[0], j[0]], DMinv.vecArray[i[ 1], j[0]], 
                                out=np.full(_de.shape, 100.0), 
                                where=(DMinv.vecArray[i[0], j[0]] * DMinv.vecArray[i[ 1], j[0]] > 0))
            mild_slope_w = (slope_w < max_slope) * (slope_w > 1/max_slope)
            mild_slope_e = (slope_e < max_slope) * (slope_e > 1/max_slope)


            Mm = self.M_pre.vecArray[i[-1],j[0]]
            Mc = self.M_pre.vecArray[i[ 0],j[0]]
            Mp = self.M_pre.vecArray[i[ 1],j[0]]
            adv_type=1
            if adv_type==0:
                adv0 = (Mc**2 * DMinv.vecArray[i[0],j[0]] - Mm**2 * DMinv.vecArray[i[-1],j[0]]) \
                    * (Mc > 0) * (Mm > 0) * dt / self.dx / (self.R * np.cos(self.yM[j[0]])) * (self.d.vecArray[i[-1],j[0]] < Dmax) * mild_slope_w
                adv1 = (Mp**2 * DMinv.vecArray[i[1],j[0]] - Mc**2 * DMinv.vecArray[i[ 0],j[0]]) \
                    * (Mc < 0) * (Mp < 0) * dt / self.dx / (self.R * np.cos(self.yM[j[0]])) * (self.d.vecArray[i[0],j[0]] < Dmax) * mild_slope_e
            elif adv_type==1:
                adv0 = (Mc**2 * DMinv.vecArray[i[0],j[0]] 
                        - Mm**2 * DMinv.vecArray[i[-1],j[0]] * (self.d.vecArray[i[-1],j[0]] < Dmax) * mild_slope_w
                        ) \
                    * (Mc > 0) * dt / self.dx / (self.R * np.cos(self.yM[j[0]])) 
                adv1 = (Mp**2 * DMinv.vecArray[i[1],j[0]] * (self.d.vecArray[i[0],j[0]] < Dmax) * mild_slope_e
                        - Mc**2 * DMinv.vecArray[i[ 0],j[0]]
                        ) \
                    * (Mc < 0) * dt / self.dx / (self.R * np.cos(self.yM[j[0]])) 
                
            _DN = self.tmpN0
            _d = np.minimum(self.d.vecArray[i[0],j[-1]], self.d.vecArray[i[0],j[0]])
            _h = np.maximum((self.h.vecArray[i[0],j[-1]] + self.h_pre.vecArray[i[0],j[-1]]) / 2,
                            (self.h.vecArray[i[0],j[ 0]] + self.h_pre.vecArray[i[0],j[ 0]]) / 2)
            _DN.vecArray[i[0],j[0]] = _d + _h
            _DN.local_to_local()

            _water = self.tmpM1 # location="left"
            _water.vecArray[i[0],j[0]] = 1
            _water.vecArray[i[0],j[0]] *= _DN.vecArray[i[-1],j[0]] > Dmin
            _water.vecArray[i[0],j[0]] *= _DN.vecArray[i[-1],j[0]] > Dmin
            _water.vecArray[i[0],j[0]] *= _DN.vecArray[i[ 0],j[1]] > Dmin
            _water.vecArray[i[0],j[0]] *= _DN.vecArray[i[ 0],j[1]] > Dmin
            _water.local_to_local()
            
            _N = self.tmpN1
            _N.vecArray[i[0],j[0]] = ((self.N_pre.vecArray[i[-1],j[0]]+self.N_pre.vecArray[i[ 0],j[1]])
                                    + (self.N_pre.vecArray[i[-1],j[1]]+self.N_pre.vecArray[i[ 0],j[0]])) / 4
            _N.vecArray[i[0],j[0]] *= _water.vecArray[i[0],j[0]]
            _N.local_to_local()

            _MND = self.tmpM2
            _MND.vecArray[i[0],j[0]] = self.M_pre.vecArray[i[0],j[0]] * _N.vecArray[i[0],j[0]] * DMinv.vecArray[i[0],j[0]]
            _MND.local_to_local()

            if adv_type==0:
                adv2 = (_MND.vecArray[i[0],j[0]] - _MND.vecArray[i[0],j[-1]])  * dt / self.dy / self.R \
                        * (_N.vecArray[i[0],j[0]] > 0) * (_N.vecArray[i[0],j[-1]] > 0) \
                        * (self.dM.vecArray[i[0],j[0]] < Dmax) * (self.dM.vecArray[i[0],j[-1]] < Dmax)
                adv3 = (_MND.vecArray[i[0],j[1]] - _MND.vecArray[i[0],j[ 0]]) * dt / self.dy / self.R \
                        * (_N.vecArray[i[0],j[1]] < 0) * (_N.vecArray[i[0],j[0]] < 0) \
                        * (self.dM.vecArray[i[0],j[0]] < Dmax) * (self.dM.vecArray[i[0],j[ 1]] < Dmax)
            elif adv_type==1:
                adv2 = (_MND.vecArray[i[0],j[0]] 
                        - _MND.vecArray[i[0],j[-1]] * (_N.vecArray[i[0],j[0]] > 0) * (_N.vecArray[i[0],j[-1]] > 0) \
                                                    * (self.dM.vecArray[i[0],j[0]] < Dmax) * (self.dM.vecArray[i[0],j[-1]] < Dmax)
                        )  * dt / self.dy / self.R 
                adv3 = (_MND.vecArray[i[0],j[1]] * (_N.vecArray[i[0],j[1]] < 0) * (_N.vecArray[i[0],j[0]] < 0) \
                                                 * (self.dM.vecArray[i[0],j[0]] < Dmax) * (self.dM.vecArray[i[0],j[ 1]] < Dmax)
                        - _MND.vecArray[i[0],j[ 0]]
                        ) * dt / self.dy / self.R 
            #self.M.vecArray[i[0], j[0]] -= (adv0 + adv1) * (tapering_factor_M==1)
            self.M.vecArray[i[0], j[0]] -= (adv0 + adv1 + adv2 + adv3) * (tapering_factor_M==1)
            self.M.local_to_local()

            # advection of N
            i, j = self.N.get_slice()
            taperI = self.taper_xh[i[0]].reshape(-1,1)
            taperJ = self.taper_yN[j[0]].reshape(1,-1)
            tapering_factor_N = taperI.dot(taperJ)

            DNinv = self.tmpN1
            _dn = self.d.vecArray[i[0],j[-1]]
            _ds = self.d.vecArray[i[0],j[ 0]]
            _d = np.minimum(self.d.vecArray[i[0],j[-1]], self.d.vecArray[i[0],j[0]])
            _h = np.maximum((self.h.vecArray[i[0],j[-1]] + self.h_pre.vecArray[i[0],j[-1]]) / 2,
                            (self.h.vecArray[i[0],j[ 0]] + self.h_pre.vecArray[i[0],j[ 0]]) / 2)
            
            _D = _d + _h
            DNinv.vecArray[i[0], j[0]] = np.divide(1, _D, out=np.zeros_like(_D), where=_D>Dmin)
            DNinv.local_to_local()

            slope_n = np.divide(DNinv.vecArray[i[0], j[0]], DNinv.vecArray[i[0], j[-1]], 
                                out=np.full(_dn.shape, 100.0), 
                                where=(DNinv.vecArray[i[0], j[0]] * DNinv.vecArray[i[0], j[-1]] > 0))
            slope_s = np.divide(DNinv.vecArray[i[0], j[0]], DNinv.vecArray[i[0], j[ 1]], 
                                out=np.full(_ds.shape, 100.0), 
                                where=(DNinv.vecArray[i[0], j[0]] * DNinv.vecArray[i[0], j[ 1]] > 0))
            mild_slope_n = (slope_n < max_slope) * (slope_n > 1/max_slope)
            mild_slope_s = (slope_s < max_slope) * (slope_s > 1/max_slope)

            Nm = self.N_pre.vecArray[i[0],j[-1]]
            Nc = self.N_pre.vecArray[i[0],j[ 0]]
            Np = self.N_pre.vecArray[i[0],j[ 1]]
            if adv_type==0:
                adv0 = (Nc**2 * DNinv.vecArray[i[0],j[0]] - Nm**2 * DNinv.vecArray[i[0],j[-1]]) \
                    * (Nc > 0) * (Nm > 0) * dt / self.dy / self.R * (self.d.vecArray[i[0],j[-1]] < Dmax) * mild_slope_n
                adv1 = (Np**2 * DNinv.vecArray[i[0],j[1]] - Nc**2 * DNinv.vecArray[i[0],j[ 0]]) \
                    * (Nc < 0) * (Np < 0) * dt / self.dy / self.R * (self.d.vecArray[i[0],j[ 1]] < Dmax) * mild_slope_s
            elif adv_type==1:
                adv0 = (Nc**2 * DNinv.vecArray[i[0],j[0]] 
                        - Nm**2 * DNinv.vecArray[i[0],j[-1]] * (self.d.vecArray[i[0],j[-1]] < Dmax) * mild_slope_n
                        ) \
                    * (Nc > 0) * dt / self.dy / self.R 
                adv1 = (Np**2 * DNinv.vecArray[i[0],j[1]] * (self.d.vecArray[i[0],j[ 1]] < Dmax) * mild_slope_s
                        - Nc**2 * DNinv.vecArray[i[0],j[ 0]]
                        ) \
                    * (Nc < 0) * dt / self.dy / self.R
            
            _DM = self.tmpM0
            _d = np.minimum(self.d.vecArray[i[-1],j[0]], self.d.vecArray[i[0],j[0]])
            _h = np.maximum((self.h.vecArray[i[-1],j[0]] + self.h_pre.vecArray[i[-1],j[0]]) / 2,
                            (self.h.vecArray[i[ 0],j[0]] + self.h_pre.vecArray[i[ 0],j[0]]) / 2)
            _DM.vecArray[i[0],j[0]] = _d + _h
            _DM.local_to_local()

            _water = self.tmpN2 # location="bottom"
            _water.vecArray[i[0],j[0]] = 1
            _water.vecArray[i[0],j[0]] *= _DN.vecArray[i[0],j[-1]] > Dmin
            _water.vecArray[i[0],j[0]] *= _DN.vecArray[i[0],j[-1]] > Dmin
            _water.vecArray[i[0],j[0]] *= _DN.vecArray[i[1],j[ 0]] > Dmin
            _water.vecArray[i[0],j[0]] *= _DN.vecArray[i[1],j[ 0]] > Dmin
            _water.local_to_local()
            
            _M = self.tmpM2
            _M.vecArray[i[0],j[0]] = ((self.M_pre.vecArray[i[0],j[-1]]+self.M_pre.vecArray[i[1],j[0]])
                                    + (self.M_pre.vecArray[i[1],j[-1]]+self.M_pre.vecArray[i[0],j[0]])) / 4
            _M.vecArray[i[0],j[0]] *= _water.vecArray[i[0],j[0]]
            _M.local_to_local()

            _NMD = self.tmpN2
            _NMD.vecArray[i[0],j[0]] = self.N_pre.vecArray[i[0],j[0]] * _M.vecArray[i[0],j[0]] * DNinv.vecArray[i[0],j[0]]
            _NMD.local_to_local()

            if adv_type==0:
                adv2 = (_NMD.vecArray[i[0],j[0]] - _NMD.vecArray[i[-1],j[0]]) * dt / self.dx / (self.R * np.cos(self.yN[j[0]])) \
                    * (_M.vecArray[i[0],j[0]] > 0) * (_M.vecArray[i[-1],j[0]] > 0) \
                    * (self.dM.vecArray[i[0],j[0]] < Dmax) * (self.dM.vecArray[i[-1],j[0]] < Dmax)
                adv3 = (_NMD.vecArray[i[1],j[0]] - _NMD.vecArray[i[ 0],j[0]]) * dt / self.dx / (self.R * np.cos(self.yN[j[0]])) \
                    * (_M.vecArray[i[1],j[0]] < 0) * (_M.vecArray[i[0],j[0]] < 0) \
                    * (self.dM.vecArray[i[0],j[0]] < Dmax) * (self.dM.vecArray[i[ 1],j[0]] < Dmax)
            elif adv_type==1:
                adv2 = (_NMD.vecArray[i[0],j[0]] 
                        - _NMD.vecArray[i[-1],j[0]] * (_M.vecArray[i[0],j[0]] > 0) * (_M.vecArray[i[-1],j[0]] > 0) \
                                                    * (self.dM.vecArray[i[0],j[0]] < Dmax) * (self.dM.vecArray[i[-1],j[0]] < Dmax)
                        ) * dt / self.dx / (self.R * np.cos(self.yN[j[0]]))
                adv3 = (_NMD.vecArray[i[1],j[0]] * (_M.vecArray[i[1],j[0]] < 0) * (_M.vecArray[i[0],j[0]] < 0) \
                                                 * (self.dM.vecArray[i[0],j[0]] < Dmax) * (self.dM.vecArray[i[ 1],j[0]] < Dmax)
                        - _NMD.vecArray[i[ 0],j[0]]
                        ) * dt / self.dx / (self.R * np.cos(self.yN[j[0]])) 
                    
            #self.N.vecArray[i[0], j[0]] -= (adv0 + adv1) * (tapering_factor_N==1)
            self.N.vecArray[i[0], j[0]] -= (adv0 + adv1 + adv2 + adv3) * (tapering_factor_N==1)
            self.N.local_to_local()


        self.hx.set_diffx(self.h, boundary_type="none")
        self.hy.set_diffy(self.h, boundary_type="none")

        if self.Nonlinear: #dM, dN change to d + h
            i, j = self.M.get_slice()
            dM_copy = self.dM.vecArray[i[0],j[0]].copy()
            self.dM.vecArray[i[0],j[0]] = (self.d.vecArray[i[0],j[0]] + self.d.vecArray[i[-1],j[0]]) / 2 \
                                        + (self.h.vecArray[i[0],j[0]] + self.h.vecArray[i[-1],j[0]]) / 2
            self.dM.local_to_local()
            i, j = self.N.get_slice()
            dN_copy = self.dN.vecArray[i[0],j[0]].copy()
            self.dN.vecArray[i[0],j[0]] = (self.d.vecArray[i[0],j[0]] + self.d.vecArray[i[0],j[-1]]) / 2 \
                                        + (self.h.vecArray[i[0],j[0]] + self.h.vecArray[i[0],j[-1]]) / 2
            self.dN.local_to_local()

        if self.Manning==0:
            # update M
            i, j = self.M.get_slice()
            Rdx = self.R * self.dx * np.cos(self.yM[j[0]])
            self.M.vecArray[i[0], j[0]] -= dt * self.gM[j[0]] \
                    * self.dM.vecArray[i[0], j[0]] / Rdx \
                    * self.hx.vecArray[i[0], j[0]]
            self.M.local_to_local()
            # update N
            i, j = self.N.get_slice()
            Rdy = self.R * self.dy
            self.N.vecArray[i[0], j[0]] -= dt * self.gN[j[0]] \
                    * self.dN.vecArray[i[0], j[0]] / Rdy \
                    * self.hy.vecArray[i[0], j[0]]
            self.N.local_to_local()
        else:
            i, j = self.M.get_slice()
            N_nw = self.N.vecArray[i[-1], j[0]]
            N_ne = self.N.vecArray[i[0], j[0]]
            N_sw = self.N.vecArray[i[-1], j[1]]
            N_se = self.N.vecArray[i[0], j[1]]
            NM = ((N_nw + N_se) + (N_ne + N_sw)) / 4 # N value at M location

            i, j = self.N.get_slice()
            M_nw = self.M.vecArray[i[0], j[-1]]
            M_ne = self.M.vecArray[i[1], j[-1]]
            M_sw = self.M.vecArray[i[0], j[0]]
            M_se = self.M.vecArray[i[1], j[0]]
            MN = ((M_nw + M_se) + (M_ne + M_sw)) / 4 # M value at N location
            # update M
            i, j = self.M.get_slice()
            Rdx = self.R * self.dx * np.cos(self.yM[j[0]])
            dMgrv = dt * self.gM[j[0]] \
                    * self.dM.vecArray[i[0], j[0]] / Rdx \
                    * self.hx.vecArray[i[0], j[0]]
            dM = self.dM.vecArray[i[0],j[0]]
            dMinv = np.divide(1, dM, out=np.zeros_like(dM), where=dM>0)
            MNabs = np.sqrt(self.M.vecArray[i[0], j[0]]**2 + NM**2)
            fric = self.gM[j[0]] * self.Manning**2 * dt / 2 * dMinv**(7/3) * MNabs
            M_pre = self.M.vecArray[i[0], j[0]]
            self.M.vecArray[i[0], j[0]] = ((1 - fric) * M_pre - dMgrv) / (1 + fric)
            self.M.local_to_local()
            # update N
            i, j = self.N.get_slice()
            Rdy = self.R * self.dy
            dNgrv = dt * self.gN[j[0]] \
                    * self.dN.vecArray[i[0], j[0]] / Rdy \
                    * self.hy.vecArray[i[0], j[0]]
            dN = self.dN.vecArray[i[0],j[0]]
            dNinv = np.divide(1, dN, out=np.zeros_like(dN), where=dN>0)
            MNabs = np.sqrt(self.N.vecArray[i[0], j[0]]**2 + MN**2)
            fric = self.gN[j[0]] * self.Manning**2 * dt / 2 * dNinv**(7/3) * MNabs
            N_pre = self.N.vecArray[i[0], j[0]]
            self.N.vecArray[i[0], j[0]] = ((1 - fric) * N_pre - dNgrv) / (1 + fric)
            self.N.local_to_local()

            if self.Nonlinear: # dM, dN change back to d without h

                i, j = self.M.get_slice()
                self.dM.vecArray[i[0], j[0]] = dM_copy
                self.dM.local_to_local()

                i, j = self.N.get_slice()
                self.dN.vecArray[i[0], j[0]] = dN_copy
                self.dN.local_to_local()
        
            

    def update_MN_by_h_adj(self, dt):
        """i, j = self.h.get_slice()
        gdMe = self.gM[j[0]] * self.dM.vecArray[i[1],j[0]] * self.M.vecArray[i[1],j[0]]
        gdMw = self.gM[j[0]] * self.dM.vecArray[i[0],j[0]] * self.M.vecArray[i[0],j[0]]
        Rdx = self.R * np.cos(self.yM[j[0]]) * self.dx 
        self.h.vecArray[i[0],j[0]] += dt * (gdMe - gdMw) / Rdx
        gdNs = self.gN[j[1]] * self.dN.vecArray[i[0],j[1]] * self.N.vecArray[i[0],j[1]]
        gdNn = self.gN[j[0]] * self.dN.vecArray[i[0],j[0]] * self.N.vecArray[i[0],j[0]]
        Rdy = self.R * self.dy
        self.h.vecArray[i[0],j[0]] += dt * (gdNs - gdNn) / Rdy
        self.h.local_to_local()"""
        """# M
        i, j = self.M.get_slice()
        Rdx = self.R * self.dx * np.cos(self.yM[j[0]])
        delta_h = dt * self.gM[j[0]] * self.dM.vecArray[i[0], j[0]] / Rdx * self.M.vecArray[i[0], j[0]]
        self.h.vecArray[i[0], j[0]] -= delta_h
        self.h.vecArray[i[-1], j[0]] += delta_h  
        # N
        i, j = self.N.get_slice()
        Rdy = self.R * self.dy
        delta_h = dt * self.gN[j[0]] * self.dN.vecArray[i[0], j[0]] / Rdy * self.N.vecArray[i[0], j[0]]
        self.h.vecArray[i[0], j[0]] -= delta_h
        self.h.vecArray[i[0], j[-1]] += delta_h 
        self.h.local_to_local()"""
        i, j = self.h.get_slice()
        iMax, jMax = self.h.da.getSizes()
        #if i.start==0:
        #    self.M.vecArray[0,:]=0
        #if i.stop==iMax:
        #    self.M.vecArray[iMax,:]=0
        gdMe = self.gM[j[0]] * self.dM.vecArray[i[1],j[0]] * self.M.vecArray[i[1],j[0]]
        gdMw = self.gM[j[0]] * self.dM.vecArray[i[0],j[0]] * self.M.vecArray[i[0],j[0]]
        if i.start==0:
            gdMw[0,:]=0
        if i.stop==iMax:
            gdMe[-1,:]=0
        Rdx = self.R * np.cos(self.yM[j[0]]) * self.dx 
        self.h.vecArray[i[0],j[0]] += dt * (gdMe - gdMw) / Rdx
        gdNs = self.gN[j[1]] * self.dN.vecArray[i[0],j[1]] * self.N.vecArray[i[0],j[1]]
        gdNn = self.gN[j[0]] * self.dN.vecArray[i[0],j[0]] * self.N.vecArray[i[0],j[0]]
        if j.start==0:
            gdNn[:,0]=0
        if j.stop==jMax:
            gdNs[:,-1]=0
        Rdy = self.R * self.dy
        self.h.vecArray[i[0],j[0]] += dt * (gdNs - gdNn) / Rdy
        self.h.local_to_local()

    def update_MN_by_Coriolis(self, dt, is_reversal=False):
        omega = 7.292e-5 # angular velocity of rotation of the Earth [rad/s]
        # dMc
        i, j = self.M.get_slice()
        taperI = self.taper_xM[i[0]].reshape(-1,1)
        taperJ = self.taper_yh[j[0]].reshape(1,-1)
        tapering_factor_M = taperI.dot(taperJ)
        N_nw = self.N.vecArray[i[-1], j[0]]
        N_ne = self.N.vecArray[i[0], j[0]]
        N_sw = self.N.vecArray[i[-1], j[1]]
        N_se = self.N.vecArray[i[0], j[1]]
        N_pre_nw = self.N_pre.vecArray[i[-1], j[0]]
        N_pre_ne = self.N_pre.vecArray[i[0], j[0]]
        N_pre_sw = self.N_pre.vecArray[i[-1], j[1]]
        N_pre_se = self.N_pre.vecArray[i[0], j[1]]
        N_c = (((N_nw + N_se) + (N_ne + N_sw)) \
             + ((N_pre_nw + N_pre_se) + (N_pre_ne + N_pre_sw))) / 8.0
        dMc = 2.0 * omega * np.sin(self.yM[j[0]]) * N_c \
            * (self.dM.vecArray[i[0],j[0]] > 0) * dt \
            * tapering_factor_M

        # dNc
        i, j = self.N.get_slice()
        taperI = self.taper_xh[i[0]].reshape(-1,1)
        taperJ = self.taper_yN[j[0]].reshape(1,-1)
        tapering_factor_N = taperI.dot(taperJ)
        M_nw = self.M.vecArray[i[0], j[-1]]
        M_ne = self.M.vecArray[i[1], j[-1]]
        M_sw = self.M.vecArray[i[0], j[0]]
        M_se = self.M.vecArray[i[1], j[0]]
        M_pre_nw = self.M_pre.vecArray[i[0], j[-1]]
        M_pre_ne = self.M_pre.vecArray[i[1], j[-1]]
        M_pre_sw = self.M_pre.vecArray[i[0], j[0]]
        M_pre_se = self.M_pre.vecArray[i[1], j[0]]
        M_c = (((M_nw + M_se) + (M_ne + M_sw)) + \
            ((M_pre_nw + M_pre_se) + (M_pre_ne + M_pre_sw))) / 8.0
        dNc = 2.0 * omega * np.sin(self.yN[j[0]]) * M_c \
            * (self.dN.vecArray[i[0],j[0]] > 0) * dt \
            * tapering_factor_N
 
        # update M and N
        if is_reversal:
            dMc *= -1
            dNc *= -1
        i, j = self.M.get_slice()
        self.M.vecArray[i[0],j[0]] -= dMc
        i, j = self.N.get_slice()
        self.N.vecArray[i[0],j[0]] += dNc
        self.M.local_to_local()
        self.N.local_to_local()
    def update_MN_by_Coriolis_adj(self, dt):
        omega = 7.292e-5 # angular velocity of rotation of the Earth [rad/s]
        # dNc
        i, j = self.N.get_slice()
        taperI = self.taper_xM[i[0]].reshape(-1,1)
        taperJ = self.taper_yh[j[0]].reshape(1,-1)
        tapering_factor_M = taperI.dot(taperJ)
        v__ = self.M.vecArray[i[0], j[0]] * (self.dM.vecArray[i[0],j[0]]>0) * np.sin(self.yM[j[0]])
        vpm = self.M.vecArray[i[1], j[-1]] * (self.dM.vecArray[i[1],j[-1]]>0) * np.sin(self.yM[j[-1]])
        vp_ = self.M.vecArray[i[1], j[0]] * (self.dM.vecArray[i[1],j[0]]>0) * np.sin(self.yM[j[0]])
        v_m = self.M.vecArray[i[0], j[-1]] * (self.dM.vecArray[i[0],j[-1]]>0) * np.sin(self.yM[j[-1]])
        v__ *= self.taper_xM[i[0]].reshape(-1,1) @ self.taper_yh[j[0]].reshape(1,-1)
        vpm *= self.taper_xM[i[1]].reshape(-1,1) @ self.taper_yh[j[-1]].reshape(1,-1)
        vp_ *= self.taper_xM[i[1]].reshape(-1,1) @ self.taper_yh[j[0]].reshape(1,-1)
        v_m *= self.taper_xM[i[0]].reshape(-1,1) @ self.taper_yh[j[-1]].reshape(1,-1)
        dNc = 2 * omega / 8 * ((v__ + vpm) + (vp_ + v_m)) * dt
        
        # dMc
        i, j = self.M.get_slice()
        taperI = self.taper_xh[i[0]].reshape(-1,1)
        taperJ = self.taper_yN[j[0]].reshape(1,-1)
        tapering_factor_N = taperI.dot(taperJ)
        v_p = self.N.vecArray[i[0],j[1]] * (self.dN.vecArray[i[0],j[1]]>0) * np.sin(self.yN[j[1]])#(self.yN[j[1]])
        vm_ = self.N.vecArray[i[-1],j[0]] * (self.dN.vecArray[i[-1],j[0]]>0) * np.sin(self.yN[j[0]])
        vmp = self.N.vecArray[i[-1],j[1]] * (self.dN.vecArray[i[-1],j[1]]>0) * np.sin(self.yN[j[1]])#(self.yN[j[1]])
        v__ = self.N.vecArray[i[0],j[0]] * (self.dN.vecArray[i[0],j[0]]>0) * np.sin(self.yN[j[0]])
        v_p *= self.taper_xh[i[0]].reshape(-1,1) @ self.taper_yN[j[1]].reshape(1,-1)
        vm_ *= self.taper_xh[i[-1]].reshape(-1,1) @ self.taper_yN[j[0]].reshape(1,-1)
        vmp *= self.taper_xh[i[-1]].reshape(-1,1) @ self.taper_yN[j[1]].reshape(1,-1)
        v__ *= self.taper_xh[i[0]].reshape(-1,1) @ self.taper_yN[j[0]].reshape(1,-1)
        dMc = 2 * omega / 8 * ((v_p + vm_) + (vmp + v__)) * dt

        i, j = self.N.get_slice()
        self.N.vecArray[i[0],j[0]] -= dNc
        self.N_pre.vecArray[i[0],j[0]] -= dNc
        i, j = self.M.get_slice()
        self.M.vecArray[i[0],j[0]] += dMc
        self.M_pre.vecArray[i[0],j[0]] += dMc
        self.M.local_to_local()
        self.N.local_to_local()
        self.M_pre.local_to_local()
        self.N_pre.local_to_local()

    def update_MN_by_Sommerfeld(self, dt):
        dt = np.abs(dt)
        i, j = self.h.get_slice()
        (iMax, jMax) = self.h.da.getSizes()
        # update M
        self.d.local_to_local()
        self.M.local_to_local()
        g_c = self.gM[j[0]]
        Rcos = self.R * np.cos(self.yM[j[0]])
        if i[0].start==0:
            d_c = self.d.vecArray[0,j[0]]
            M_w = self.M_pre.vecArray[0,j[0]]
            M_e = self.M_pre.vecArray[1,j[0]]
            dM = M_e - M_w
            self.M.vecArray[0,j[0]] += np.sqrt(g_c * d_c) \
                / (Rcos * self.dx) * (d_c > 0) * dt * dM
        if i[0].stop==iMax:
            d_c = self.d.vecArray[iMax-1,j[0]]
            M_e = self.M_pre.vecArray[iMax,j[0]]
            M_w = self.M_pre.vecArray[iMax-1,j[0]]
            f = np.sqrt(g_c * d_c) / (Rcos * self.dx) * (d_c > 0) * dt
            dM = M_w - M_e
            #self.M.vecArray[iMax,j[0]] += np.sqrt(g_c * d_c) \
            #    / (Rcos * self.dx) * (d_c > 0) * dt * dM
            self.M.vecArray[iMax,j[0]] += f * dM # (M_w - M_e)
        self.M.local_to_local()
        # update N
        if j[0].start==0:
            g_c = self.gN[0]
            Rcos = self.R * np.cos(self.yM[0])
            d_c = self.d.vecArray[i[0],0]
            N_c = self.N_pre.vecArray[i[0],0]
            N_s = self.N_pre.vecArray[i[0],1]
            dN = N_s * np.cos(self.yN[1]) \
                - N_c * np.cos(self.yN[0])
            self.N.vecArray[i[0],0] += np.sqrt(g_c * d_c) \
                / (Rcos * self.dy) * (d_c > 0) * dt * dN
        if j[0].stop==jMax:
            g_c = self.gN[jMax]
            Rcos = self.R * np.cos(self.yM[jMax-1])
            d_c = self.d.vecArray[i[0], jMax-1]
            N_c = self.N_pre.vecArray[i[0],jMax]
            N_n = self.N_pre.vecArray[i[0],jMax-1]
            dN = N_n * np.cos(self.yN[jMax-1]) \
                - N_c * np.cos(self.yN[jMax])
            self.N.vecArray[i[0],jMax] += np.sqrt(g_c * d_c) \
                / (Rcos * self.dy) * (d_c > 0) * dt * dN
        self.N.local_to_local()
    def update_MN_by_Sommerfeld_adj(self, dt):
        i, j = self.h.get_slice()
        (iMax, jMax) = self.h.da.getSizes()
        # update M
        self.d.local_to_local()    
        self.M.local_to_local()    
        g_c = self.gM[j[0]]
        Rcos = self.R * np.cos(self.yM[j[0]])
        reversed_sign = False
        if i[0].start==0:
            d_c = self.d.vecArray[0,j[0]]
            M_c = self.M.vecArray[0,j[0]]
            dM = np.sqrt(g_c * d_c) / (Rcos * self.dx) \
                * (d_c > 0) * dt * M_c
            if reversed_sign:
                self.M_pre.vecArray[0,j[0]] += dM # reversed sign
                self.M_pre.vecArray[1,j[0]] -= dM # reversed sign
            else:
                self.M_pre.vecArray[0,j[0]] -= dM
                self.M_pre.vecArray[1,j[0]] += dM
        if i[0].stop==iMax:
            #print(i[0].stop, i[0], self.dx, Rcos, dt)
            d_c = self.d.vecArray[iMax-1,j[0]]
            M_c = self.M.vecArray[iMax-0,j[0]]
            #dM = np.sqrt(g_c * d_c) / (Rcos * self.dx) \
            #    * (d_c > 0) * dt * M_c
            #self.M.vecArray[iMax,j[0]] -= dM
            #self.M.vecArray[iMax-1,j[0]] += dM
            f = np.sqrt(g_c * d_c) / (Rcos * self.dx) * (d_c > 0) * dt
            delta_M = f * M_c
            # The error happens when step > 438.
            #self.M.vecArray[iMax,j[0]] -= f * M_c
            #self.M.vecArray[iMax-1,j[0]] += f * M_c
            # test case
            if reversed_sign:
                self.M_pre.vecArray[iMax-1,j[0]] -= delta_M # reversed sign
                self.M_pre.vecArray[iMax,j[0]] += delta_M # reversed sign
            else:
                self.M_pre.vecArray[iMax-1,j[0]] += delta_M
                self.M_pre.vecArray[iMax-0,j[0]] -= delta_M
        self.M.local_to_local()
            
        # update N
        if j[0].start==0:
            g_c = self.gN[0]
            Rcos = self.R * np.cos(self.yM[0])
            d_c = self.d.vecArray[i[0],0]
            N_c = self.N.vecArray[i[0],0]
            dN = np.sqrt(g_c * d_c) / (Rcos * self.dy) \
                * (d_c > 0) * dt * N_c
            if reversed_sign:
                self.N_pre.vecArray[i[0],0] += dN * np.cos(self.yN[0]) # reversed sign
                self.N_pre.vecArray[i[0],1] -= dN * np.cos(self.yN[1]) # reversed sign
            else:
                self.N_pre.vecArray[i[0],0] -= dN * np.cos(self.yN[0])
                self.N_pre.vecArray[i[0],1] += dN * np.cos(self.yN[1])
            
        if j[0].stop==jMax:
            g_c = self.gN[jMax]
            Rcos = self.R * np.cos(self.yM[jMax-1])
            d_c = self.d.vecArray[i[0], jMax-1]
            N_c = self.N.vecArray[i[0],jMax]
            dN = np.sqrt(g_c * d_c) / (Rcos * self.dy) \
                * (d_c > 0) * dt * N_c
            if reversed_sign:
                self.N_pre.vecArray[i[0],jMax] += dN * np.cos(self.yN[jMax]) # reversed sign
                self.N_pre.vecArray[i[0],jMax-1] -= dN * np.cos(self.yN[jMax-1]) # reversed sign
            else:
                self.N_pre.vecArray[i[0],jMax] -= dN * np.cos(self.yN[jMax])
                self.N_pre.vecArray[i[0],jMax-1] += dN * np.cos(self.yN[jMax-1])
        self.N.local_to_local()

    def update_MN_by_sponge(self, dt):
        dt = np.abs(dt)
        # hyperbolic damping fanction
        # Cruz, 横木, 磯部, 渡辺 (1993) 非線形波動方程式に対する無反射境界条件について, 海岸工学論文集, 40, 46-50
        r = self.hyperbolic_r
        cd = r * self.damping_factor / (2 * (np.sinh(r) - r))  # correction after Cruz et al. (1993)

        # update M
        i, j = self.M.get_slice(edge=True)
        taperI = self.taper_xM[i[0]].reshape(-1,1)
        taperJ = self.taper_yh[j[0]].reshape(1,-1)
        tapering_factor_M = cd * (np.cosh(r * (1 - taperI.dot(taperJ))) - 1)
        
        M_c = self.M.vecArray[i[0], j[0]]
        #d_w = self.d.vecArray[i[-1], j[0]]
        #d_e = self.d.vecArray[i[0], j[0]]
        #d_c = np.maximum((d_w + d_e) / 2, 10)
        d_c = np.maximum(self.dM.vecArray[i[0], j[0]], 10)
        g_c = self.gM[j[0]]
        dMc = tapering_factor_M * np.sqrt(g_c / d_c) \
              * M_c * (d_c > 0) * dt
        self.M.vecArray[i[0], j[0]] -= dMc
        self.M.local_to_local()
        
        # update N
        i, j = self.N.get_slice(edge=True)
        taperI = self.taper_xh[i[0]].reshape(-1,1)
        taperJ = self.taper_yN[j[0]].reshape(1,-1)
        tapering_factor_N = cd * (np.cosh(r * (1 - taperI.dot(taperJ))) - 1)

        N_c = self.N.vecArray[i[0], j[0]]
        #d_n = self.d.vecArray[i[0], j[-1]]
        #d_s = self.d.vecArray[i[0], j[0]]
        #d_c = np.maximum((d_n + d_s) / 2, 10)
        d_c = np.maximum(self.dN.vecArray[i[0], j[0]], 10)
        g_c = self.gN[j[0]]
        dNc = tapering_factor_N * np.sqrt(g_c / d_c) \
              * N_c * (d_c > 0) * dt
        self.N.vecArray[i[0], j[0]] -= dNc
        self.N.local_to_local()

    def update_b_by_h(self):
        i, j = self.h.get_slice()
        iMax, jMax = self.shape
        # boundary copy
        self.h.set_boundary(boundary_type="copy")
        self.d.set_boundary(boundary_type="zero")
        g_c = self.gM[j[0]]
        g_n = self.gN[j[0]]
        g_s = self.gN[j[1]]
        cos_c = np.cos(self.yM[j[0]])
        cos_n = np.cos(self.yN[j[0]])
        cos_s = np.cos(self.yN[j[1]])
        h_c = self.h.vecArray[i[0],j[0]]
        h_w = self.h.vecArray[i[-1],j[0]]
        h_e = self.h.vecArray[i[1],j[0]]
        h_n = self.h.vecArray[i[0],j[-1]]
        h_s = self.h.vecArray[i[0],j[1]]
        d_c = self.d.vecArray[i[0],j[0]]
        d_w = self.d.vecArray[i[-1],j[0]]
        d_e = self.d.vecArray[i[1],j[0]]
        d_n = self.d.vecArray[i[0],j[-1]]
        d_s = self.d.vecArray[i[0],j[1]]
        dcw = (d_w + d_c) / 2 * (d_c > 0) * (d_w > 0)
        dce = (d_e + d_c) / 2 * (d_c > 0) * (d_e > 0)
        dcn = (d_n + d_c) / 2 * (d_c > 0) * (d_n > 0)
        dcs = (d_s + d_c) / 2 * (d_c > 0) * (d_s > 0)
        L_w = g_c * dcw * d_c / (3 * (self.R * cos_c * self.dx)**2)
        L_e = g_c * dce * d_c / (3 * (self.R * cos_c * self.dx)**2)
        L_n = g_n * dcn * d_c / (3 * (self.R * self.dy)**2) * cos_n / cos_c
        L_s = g_s * dcs * d_c / (3 * (self.R * self.dy)**2) * cos_s / cos_c
        L_c = -((L_w + L_e) + (L_n + L_s))
        self.b.vecArray[i[0],j[0]] = (L_w * h_w + L_e * h_e) \
                                    + (L_n * h_n + L_s * h_s) + L_c * h_c
        self.b.local_to_local()
        self.d.set_boundary(boundary_type="copy")
    def update_b_by_h_adj(self):
        i, j = self.b.get_slice()
        iMax, jMax = self.shape
        # boundary copy
        self.b.set_boundary(boundary_type="copy")
        self.d.set_boundary(boundary_type="zero")
        g_c = self.gM[j[0]]
        g_n = self.gN[j[0]]
        g_s = self.gN[j[1]]
        cos_c = np.cos(self.yM[j[0]])
        cos_n = np.cos(self.yM[j[0]]+self.dy)
        cos_s = np.cos(self.yM[j[0]]-self.dy)
        cos_cn = np.cos(self.yN[j[0]])
        cos_cs = np.cos(self.yN[j[1]])
        b_c = self.b.vecArray[i[0],j[0]]
        b_w = self.b.vecArray[i[-1],j[0]]
        b_e = self.b.vecArray[i[1],j[0]]
        b_n = self.b.vecArray[i[0],j[-1]]
        b_s = self.b.vecArray[i[0],j[1]]
        d_c = self.d.vecArray[i[0],j[0]]
        d_w = self.d.vecArray[i[-1],j[0]]
        d_e = self.d.vecArray[i[1],j[0]]
        d_n = self.d.vecArray[i[0],j[-1]]
        d_s = self.d.vecArray[i[0],j[1]]
        dcw = (d_w + d_c) / 2 * (d_c > 0) * (d_w > 0)
        dce = (d_e + d_c) / 2 * (d_c > 0) * (d_e > 0)
        dcn = (d_n + d_c) / 2 * (d_c > 0) * (d_n > 0)
        dcs = (d_s + d_c) / 2 * (d_c > 0) * (d_s > 0)
        L_w = g_c * dcw * d_c / (3 * (self.R * cos_c * self.dx)**2)
        L_e = g_c * dce * d_c / (3 * (self.R * cos_c * self.dx)**2)
        L_n = g_n * dcn * d_c / (3 * (self.R * self.dy)**2) * cos_cn / cos_c
        L_s = g_s * dcs * d_c / (3 * (self.R * self.dy)**2) * cos_cs / cos_c
        L_c = -((L_w + L_e) + (L_n + L_s))
        L_w = g_c * dcw * d_w / (3 * (self.R * cos_c * self.dx)**2)
        L_e = g_c * dce * d_e / (3 * (self.R * cos_c * self.dx)**2)
        L_n = g_n * dcn * d_n / (3 * (self.R * self.dy)**2) * cos_cn / cos_n
        L_s = g_s * dcs * d_s / (3 * (self.R * self.dy)**2) * cos_cs / cos_s
        self.h.vecArray[i[0],j[0]] += (L_w * b_w + L_e * b_e) \
                                    + (L_n * b_n + L_s * b_s) + L_c * b_c
        self.h.local_to_local()
        self.b.vecArray[i[0],j[0]] = 0
        self.b.local_to_local()
        self.d.set_boundary(boundary_type="copy")

    def update_Phai_by_b(self):
        self.b.local_to_global()
        #self.setup_ksp(is_adjoint=False)
        self.ksp.solve(self.b.globalVec, self.Phai.globalVec)
        self.Phai.global_to_local()
        i, j = self.b.get_slice()
        self.b.vecArray[i[0],j[0]] = 0
        self.b.local_to_local()
    def update_Phai_by_b_adj(self):
        self.Phai.local_to_global()
        #self.setup_ksp(is_adjoint=True)
        self.ksp.solve(self.Phai.globalVec, self.b0.globalVec)
        self.b0.global_to_local()
        i, j = self.b0.get_slice()
        self.b.vecArray[i[0],j[0]] += self.b0.vecArray[i[0],j[0]]
        self.b.local_to_local()
        self.Phai.vecArray[i[0],j[0]] = 0
        self.Phai.local_to_local()
        
    def update_Phai_by_tapering(self):
        i, j = self.Phai.get_slice()
        taperI = self.taper_xh[i[0]].reshape(-1,1)
        taperJ = self.taper_yh[j[0]].reshape(1,-1)
        tapering_factor = taperI.dot(taperJ)
        self.Phai.vecArray[i[0],j[0]] *= tapering_factor
        self.Phai.local_to_local()

    def update_MN_by_Phai(self, dt):
        # update M
        i, j = self.M.get_slice()
        Rdx = self.R * self.dx * np.cos(self.yM[j[0]])
        d_c = self.dM.vecArray[i[0],j[0]]
        Phai_e = self.Phai.vecArray[i[0],j[0]]
        Phai_w = self.Phai.vecArray[i[-1],j[0]]
        self.M.vecArray[i[0], j[0]] += dt * d_c / Rdx \
                * (Phai_e - Phai_w)
        self.M.local_to_local()
        # update N
        i, j = self.N.get_slice()
        Rdy = self.R * self.dy
        d_c = self.dN.vecArray[i[0],j[0]]
        Phai_s = self.Phai.vecArray[i[0],j[0]]
        Phai_n = self.Phai.vecArray[i[0],j[-1]]
        self.N.vecArray[i[0], j[0]] += dt * d_c / Rdy \
                * (Phai_s - Phai_n)
        self.N.local_to_local()
    def update_MN_by_Phai_adj(self, dt):
        # update M
        i, j = self.Phai.get_slice()
        Rdx = self.R * self.dx * np.cos(self.yM[j[0]])
        Rdy = self.R * self.dy

        d_e = self.dM.vecArray[i[0],j[0]]
        d_w = self.dM.vecArray[i[1],j[0]]
        d_n = self.dN.vecArray[i[0],j[0]]
        d_s = self.dN.vecArray[i[0],j[1]]
        M_e = self.M.vecArray[i[0],j[0]]
        M_w = self.M.vecArray[i[1],j[0]]
        N_n = self.N.vecArray[i[0],j[0]]
        N_s = self.N.vecArray[i[0],j[1]]

        self.Phai.vecArray[i[0], j[0]] += \
            dt / Rdx * (d_e * M_e - d_w * M_w) \
            + dt / Rdy * (d_n * N_n - d_s * N_s)

        self.Phai.local_to_local()

    def update_AMBN_by_MN(self, dt):
        i, j = self.h.get_slice()
        M_e = self.M.vecArray[i[1], j[0]]
        M_w = self.M.vecArray[i[0], j[0]]
        N_s = self.N.vecArray[i[0], j[1]]
        N_n = self.N.vecArray[i[0], j[0]]
        d_c = self.d.vecArray[i[0], j[0]]
        cos_s = np.cos(self.yN[j[1]])
        cos_n = np.cos(self.yN[j[0]])
        cos_c = np.cos(self.yM[j[0]])
        Rdx = self.R * self.dx * np.cos(self.yM[j[0]])
        inv_Rdx = 1.0 / Rdx
        inv_Rdy_s = (cos_s / cos_c) / (self.R * self.dy)
        inv_Rdy_n = (cos_n / cos_c) / (self.R * self.dy)
        
        if self.Nonlinear: # flux limitter 2D
            h_c = self.h.vecArray[i[0],j[0]]
            D = d_c + h_c
            Rdy = self.R * self.dy
            WaterVolume = D * Rdx * Rdy
            OutVolume = (M_e * (M_e>0) - M_w * (M_w<0)) * Rdy + (N_s * (N_s>0) - N_n * (N_n<0)) * Rdx
            limit = np.divide(WaterVolume, OutVolume, out=np.zeros_like(WaterVolume), where=OutVolume>0)
            limit[limit<0]=0
            limit[limit>1]=1
            limit_w = np.copy(limit)
            limit_e = np.copy(limit)
            limit_n = np.copy(limit)
            limit_s = np.copy(limit)
            limit_w[M_w>=0]=1
            limit_e[M_e<=0]=1
            limit_n[N_n>=0]=1
            limit_s[N_s<=0]=1
            M_w *= limit_w
            M_e *= limit_e
            N_n *= limit_n
            N_s *= limit_s
            self.M.local_to_local()
            self.N.local_to_local()

        # h can change if d_c > 0
        self.AMBN.vecArray[i[0], j[0]] = (d_c > 0) \
            * ((M_w - M_e) * inv_Rdx + (N_n * inv_Rdy_n - N_s * inv_Rdy_s))
        self.AMBN.local_to_local()
        # M and N must be zero at coast
        i,j=self.M.get_slice()
        self.M.vecArray[i[0],j[0]] *= self.d.vecArray[i[ 0],j[0]] > 0
        self.M.vecArray[i[0],j[0]] *= self.d.vecArray[i[-1],j[0]] > 0
        self.M.local_to_local()
        i,j=self.N.get_slice()
        self.N.vecArray[i[0],j[0]] *= self.d.vecArray[i[0],j[ 0]] > 0
        self.N.vecArray[i[0],j[0]] *= self.d.vecArray[i[0],j[-1]] > 0
        self.N.local_to_local()

    def update_AMBN_by_MN_adj(self, dt):
        """i, j = self.h.get_slice()
        d_c = self.d.vecArray[i[0], j[0]]
        cos_s = np.cos(self.yN[j[1]])
        cos_n = np.cos(self.yN[j[0]])
        cos_c = np.cos(self.yM[j[0]])
        Rdx = self.R * self.dx * np.cos(self.yM[j[0]])
        inv_Rdx = 1.0 / Rdx
        inv_Rdy_s = (cos_s / cos_c) / (self.R * self.dy)
        inv_Rdy_n = (cos_n / cos_c) / (self.R * self.dy)
        AMBN_c = self.AMBN.vecArray[i[0], j[0]]
        self.M.vecArray[i[0], j[0]] += (d_c > 0) * inv_Rdx * AMBN_c
        self.M.vecArray[i[1], j[0]] -= (d_c > 0) * inv_Rdx * AMBN_c
        self.N.vecArray[i[0], j[0]] += (d_c > 0) * inv_Rdy_n * AMBN_c
        self.N.vecArray[i[0], j[1]] -= (d_c > 0) * inv_Rdy_s * AMBN_c
        self.M.local_to_local()
        self.N.local_to_local()"""
        # M
        i, j = self.M.get_slice(edge=True)
        iMax, jMax = self.M.da.getSizes()
        AMBN_w = self.AMBN.vecArray[i[-1], j[0]] * (self.d.vecArray[i[-1], j[0]] > 0)
        AMBN_e = self.AMBN.vecArray[i[0], j[0]] * (self.d.vecArray[i[0], j[0]] > 0)
        if i.start==0:
            AMBN_w[0,:]=0
        if i.stop==iMax:
            AMBN_e[-1,:]=0
        cos_c = np.cos(self.yM[j[0]])
        self.M.vecArray[i[0], j[0]] += \
            (AMBN_e - AMBN_w) / (self.R * self.dx * cos_c)
        self.M.local_to_local()
        # N
        i, j = self.N.get_slice(edge=True)
        AMBN_n = self.AMBN.vecArray[i[0], j[-1]] * (self.d.vecArray[i[0], j[-1]] > 0)
        AMBN_s = self.AMBN.vecArray[i[0], j[0]] * (self.d.vecArray[i[0], j[0]] > 0)
        if j.start==0:
            AMBN_n[:,0]=0
        if j.stop==jMax:
            AMBN_s[:,-1]=0
        cos_n = np.cos(self.yN[j[0]] + self.dy / 2)
        cos_s = np.cos(self.yN[j[0]] - self.dy / 2)
        cos_c = np.cos(self.yN[j[0]])
        self.N.vecArray[i[0], j[0]] += \
            (AMBN_s / cos_s - AMBN_n / cos_n) * cos_c / (self.R * self.dy)
        self.N.local_to_local()
        # AMBN
        i, j = self.h.get_slice()
        self.AMBN.vecArray[i[0], j[0]] = 0
        self.AMBN.local_to_local()
        
    def update_h_by_AMBN(self, dt, is_intermediate=False):
        i, j = self.h.get_slice()
        ij = (i[0], j[0])
        #taperI = self.taper_xh[i[0]].reshape(-1,1)
        #taperJ = self.taper_yh[j[0]].reshape(1,-1)
        #tapering_factor = taperI.dot(taperJ)
        d_c = self.d.vecArray[ij]
        z_c = self.z.vecArray[ij]
        h_pre_c = self.h_pre.vecArray[ij]
        z_pre_c = self.z_pre.vecArray[ij]
        AMBN_c = self.AMBN.vecArray[ij]
        comp_c = self.comp.vecArray[ij]
        if is_intermediate:
            self.h_star.vecArray[ij] = (d_c > 0) \
                * (h_pre_c - z_pre_c       + AMBN_c * dt * comp_c) #* tapering_factor
            self.h_star.local_to_local()
        else: 
            self.h.vecArray[ij] = (d_c > 0) \
                * (h_pre_c - z_pre_c + z_c + AMBN_c * dt * comp_c) #* tapering_factor
            self.h.local_to_local()
    def update_h_by_AMBN_adj(self, dt, is_intermediate=False):
        i, j = self.h.get_slice()
        ij = (i[0], j[0])
        #taperI = self.taper_xh[i[0]].reshape(-1,1)
        #taperJ = self.taper_yh[j[0]].reshape(1,-1)
        #tapering_factor = taperI.dot(taperJ)
        d_c = self.d.vecArray[ij]
        if is_intermediate:
            h_c = self.h_star.vecArray[ij] * (d_c > 0)
        else:
            h_c = self.h.vecArray[ij] * (d_c > 0)
        comp_c = self.comp.vecArray[ij]
        
        self.h_pre.vecArray[ij] += h_c #* tapering_factor
        self.z_pre.vecArray[ij] -= h_c #* tapering_factor
        self.AMBN.vecArray[ij] += dt * comp_c * h_c #* tapering_factor
        self.h_pre.local_to_local()
        self.z_pre.local_to_local()
        self.AMBN.local_to_local()
        if is_intermediate:
            self.h_star.vecArray[ij] = 0
            self.h_star.local_to_local()
        else: 
            self.z.vecArray[ij] += h_c #* tapering_factor
            self.z.local_to_local()
            self.h.vecArray[ij] = 0
            self.h.local_to_local()
        
    def update_h_by_AMBN_with_SAL(self, dt):            
        self.update_h_by_AMBN(dt, is_intermediate=True)
        h_star_coarse = self.dmHierarchy.get_coarse(self.h_star)
        if self.rank==0:
            z_coarse = self.solid_earth.get_z_coarse_by_SAL(h_star_coarse, self.CLV)
            #z_coarse = h_star_coarse * 0.01######## only for debug ##############
        else: 
            z_coarse = None
        self.dmHierarchy.to_fine(z_coarse, self.z)
        ##### only for debug #####
        #i, j = self.h.get_slice()
        #ij = (i[0], j[0])
        #self.z.vecArray[ij] = 0.01 * self.h_star.vecArray[ij]
        self.z.local_to_local()
        ##########################
        self.update_h_by_AMBN(dt, is_intermediate=False)
    def update_h_by_AMBN_with_SAL_adj(self, dt):            
        self.update_h_by_AMBN_adj(dt, is_intermediate=False)
        delta_z_coarse = self.dmHierarchy.get_coarse(self.z)
        if self.rank==0:
            z_coarse = self.z_coarse_pre + delta_z_coarse
            self.z_coarse_pre[:] = z_coarse
            delta_h_star_coarse = self.solid_earth.get_z_coarse_by_SAL(z_coarse, self.CLV)
            #delta_h_star_coarse = z_coarse * 0.01######## only for debug ##############
            h_star_coarse = self.h_star_coarse_pre + delta_h_star_coarse
            self.z_coarse_pre[:] = 0
        else: 
            h_star_coarse = None
        i, j = self.h.get_slice()
        ij = (i[0], j[0])
        self.z.vecArray[ij]=0
        self.z.local_to_local()
        self.dmHierarchy.to_fine(h_star_coarse, self.z)
        self.h_star.vecArray[ij] += self.z.vecArray[ij]
        self.h_star.local_to_local()
        ##### only for debug #####
        #i, j = self.h.get_slice()
        #ij = (i[0], j[0])
        #self.h_star.vecArray[ij] += 0.01 * self.z.vecArray[ij]
        #self.z.vecArray[ij]=0
        #self.h_star.local_to_local()
        #self.z.local_to_local()
        ##########################
        #self.h_star.local_to_local()
        if self.rank==0:
            self.h_star_coarse_pre[:] = 0
        self.update_h_by_AMBN_adj(dt, is_intermediate=True)

    def update_hmax_by_h(self):
        i, j = self.h.get_slice()
        ij = (i[0], j[0])
        self.hmax.vecArray[ij] = np.maximum(self.hmax.vecArray[ij], self.h.vecArray[ij])
        self.hmax.local_to_local()

    def copy_to_pre(self):
       i, j = self.h.get_slice()
       self.h_pre.vecArray[i[0],j[0]] = self.h.vecArray[i[0],j[0]]
       self.z_pre.vecArray[i[0],j[0]] = self.z.vecArray[i[0],j[0]]
       (i0, i1), (j0, j1) = self.M.da.getRanges()
       self.M_pre.vecArray[i0:i1,j0:j1] = self.M.vecArray[i0:i1,j0:j1]
       (i0, i1), (j0, j1) = self.N.da.getRanges()
       self.N_pre.vecArray[i0:i1,j0:j1] = self.N.vecArray[i0:i1,j0:j1]
       self.h_pre.local_to_local()
       self.z_pre.local_to_local()
       self.M_pre.local_to_local()
       self.N_pre.local_to_local()
    def copy_to_pre_adj(self):
       (i0, i1), (j0, j1) = self.h.da.getRanges()
       self.h.vecArray[i0:i1,j0:j1] += self.h_pre.vecArray[i0:i1,j0:j1]
       self.z.vecArray[i0:i1,j0:j1] += self.z_pre.vecArray[i0:i1,j0:j1]
       self.h_pre.vecArray[i0:i1,j0:j1] = 0
       self.z_pre.vecArray[i0:i1,j0:j1] = 0
       (i0, i1), (j0, j1) = self.M.da.getRanges()
       self.M.vecArray[i0:i1,j0:j1] += self.M_pre.vecArray[i0:i1,j0:j1]
       self.M_pre.vecArray[i0:i1,j0:j1] = 0
       (i0, i1), (j0, j1) = self.N.da.getRanges()
       self.N.vecArray[i0:i1,j0:j1] += self.N_pre.vecArray[i0:i1,j0:j1]
       self.N_pre.vecArray[i0:i1,j0:j1] = 0
       
       self.h.local_to_local()
       self.z.local_to_local()
       self.M.local_to_local()
       self.N.local_to_local()
       self.h_pre.local_to_local()
       self.z_pre.local_to_local()
       self.M_pre.local_to_local()
       self.N_pre.local_to_local()

    def set_Poisson(self, ksp, MatPc, Mat):
        da = self.Phai.da
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        (i0, i1), (j0, j1) = da.getRanges()
        (iMax, jMax) = da.getSizes()
        #rows, cols, vals = [],[],[]
        #Istart, Iend = Mat.getOwnershipRange()
        #ranges = da.getOwnershipRanges()
        #rankI = self.rank % ranges[0].size
        #rankJ = self.rank // ranges[0].size
        i, j = self.d.get_slice()

        d_c = self.d.vecArray[i[0],j[0]]
        d_w = self.d.vecArray[i[-1],j[0]]
        d_e = self.d.vecArray[i[1],j[0]] 
        d_n = self.d.vecArray[i[0],j[-1]]
        d_s = self.d.vecArray[i[0],j[1]] 
        dcw = (d_c + d_w) / 2
        dce = (d_c + d_e) / 2
        dcn = (d_c + d_n) / 2
        dcs = (d_c + d_s) / 2
        
        yM_array = np.ones(j1-j0)
        yN0array = np.ones(j1-j0)
        yN1array = np.ones(j1-j0)
        yM_array[:self.yM[j[0]].size ] = self.yM[j[0]]
        yN0array[:self.yM[j[0]].size ] = self.yN[j[0]]
        yN1array[:self.yM[j[0]].size ] = self.yN[j[1]]
        cos_c = np.cos(yM_array)
        cos_cn = np.cos(yN0array)
        cos_cs = np.cos(yN1array)
        K_w = d_c * dcw / (3 * (self.R * cos_c * self.dx)**2)
        K_e = d_c * dce / (3 * (self.R * cos_c * self.dx)**2)
        K_n = d_c * dcn / (3 * (self.R * self.dy)**2) * cos_cn / cos_c
        K_s = d_c * dcs / (3 * (self.R * self.dy)**2) * cos_cs / cos_c
        #K_w = d_c **2 / (3 * (self.R * cos_c * self.dx)**2)
        #K_e = d_c **2 / (3 * (self.R * cos_c * self.dx)**2)
        #K_n = d_c **2 / (3 * (self.R * self.dy)**2) * cos_cn / cos_c
        #K_s = d_c **2 / (3 * (self.R * self.dy)**2) * cos_cs / cos_c
        for j in range(j0, j1):
            for i in range(i0, i1):
                row.index = (i, j)
                Kw = K_w[i-i0, j-j0] 
                Ke = K_e[i-i0, j-j0] 
                Kn = K_n[i-i0, j-j0] 
                Ks = K_s[i-i0, j-j0] 
                # j-1
                if j > 0 and self.d.vecArray[i,j-1] > 0:
                    col.index = (i, j-1)
                    Mat.setValueStencil(row, col, Kn)
                # i-1
                if i > 0 and self.d.vecArray[i-1,j]>0:
                    col.index = (i-1, j)
                    Mat.setValueStencil(row, col, Kw)
                # center
                val = -1 #- ((KTw + KTe) + (KTn + KTs))
                
                isFixBoundary = True
                if isFixBoundary:
                    if j>0      and self.d.vecArray[i,j-1]>0: val -= Kn
                    if i>0      and self.d.vecArray[i-1,j]>0: val -= Kw
                    if i<iMax-1 and self.d.vecArray[i+1,j]>0: val -= Ke
                    if j<jMax-1 and self.d.vecArray[i,j+1]>0: val -= Ks
                col.index = (i, j)
                Mat.setValueStencil(row, col, val)
                # i+1
                if i < iMax-1 and self.d.vecArray[i+1,j] > 0:
                    col.index = (i+1, j)
                    Mat.setValueStencil(row, col, Ke)
                # j+1
                if j < jMax-1 and self.d.vecArray[i,j+1] > 0:
                    col.index = (i, j+1)
                    Mat.setValueStencil(row, col, Ks)    
        Mat.assemble()
        return 0
    def setup_Poisson_Mat(self):
        da = self.Phai.da
        #row = PETSc.Mat.Stencil()
        #col = PETSc.Mat.Stencil()
        (i0, i1), (j0, j1) = da.getRanges()
        (iMax, jMax) = da.getSizes()
        self.Mat = PETSc.Mat()
        self.Mat.create(PETSc.COMM_WORLD)
        self.Mat.setSizes([iMax * jMax, iMax * jMax])
        self.Mat.setType("aij")
        self.Mat.setPreallocationNNZ(5)
        #da = self.Phai.da
        #row = PETSc.Mat.Stencil()
        #col = PETSc.Mat.Stencil()
        #(i0, i1), (j0, j1) = da.getRanges()
        #(iMax, jMax) = da.getSizes()
        #rows, cols, vals = [],[],[]
        Istart, Iend = self.Mat.getOwnershipRange()
        #ranges = da.getOwnershipRanges()
        #rankI = self.rank % ranges[0].size
        #rankJ = self.rank // ranges[0].size
        i, j = self.d.get_slice()

        d_c = self.d.vecArray[i[0],j[0]]
        d_w = self.d.vecArray[i[-1],j[0]]
        d_e = self.d.vecArray[i[1],j[0]] 
        d_n = self.d.vecArray[i[0],j[-1]]
        d_s = self.d.vecArray[i[0],j[1]] 
        dcw = (d_c + d_w) / 2
        dce = (d_c + d_e) / 2
        dcn = (d_c + d_n) / 2
        dcs = (d_c + d_s) / 2
        
        yM_array = np.ones(j1-j0)
        yN0array = np.ones(j1-j0)
        yN1array = np.ones(j1-j0)
        yM_array[:self.yM[j[0]].size ] = self.yM[j[0]]
        yN0array[:self.yM[j[0]].size ] = self.yN[j[0]]
        yN1array[:self.yM[j[0]].size ] = self.yN[j[1]]
        cos_c = np.cos(yM_array)
        cos_cn = np.cos(yN0array)
        cos_cs = np.cos(yN1array)
        K_w = d_c * dcw / (3 * (self.R * cos_c * self.dx)**2)
        K_e = d_c * dce / (3 * (self.R * cos_c * self.dx)**2)
        K_n = d_c * dcn / (3 * (self.R * self.dy)**2) * cos_cn / cos_c
        K_s = d_c * dcs / (3 * (self.R * self.dy)**2) * cos_cs / cos_c
        for j in range(j0, j1):
            for i in range(i0, i1):
                #row.index = (i, j)
                row = i + iMax * j
                Kw = K_w[i-i0, j-j0] 
                Ke = K_e[i-i0, j-j0] 
                Kn = K_n[i-i0, j-j0] 
                Ks = K_s[i-i0, j-j0] 
                # j-1
                if j > 0 and self.d.vecArray[i,j-1] > 0:
                    #col.index = (i, j-1)
                    col = i + iMax * (j - 1)
                    #Mat.setValueStencil(row, col, Kn)
                    self.Mat[row, col] = Kn
                # i-1
                if i > 0 and self.d.vecArray[i-1,j]>0:
                    #col.index = (i-1, j)
                    col = i - 1 + iMax * j
                    #Mat.setValueStencil(row, col, Kw)
                    self.Mat[row, col] = Kw
                # center
                val = -1 #- ((KTw + KTe) + (KTn + KTs))
                
                isFixBoundary = True
                if isFixBoundary:
                    if j>0      and self.d.vecArray[i,j-1]>0: val -= Kn
                    if i>0      and self.d.vecArray[i-1,j]>0: val -= Kw
                    if i<iMax-1 and self.d.vecArray[i+1,j]>0: val -= Ke
                    if j<jMax-1 and self.d.vecArray[i,j+1]>0: val -= Ks
                #col.index = (i, j)
                col = i + iMax * j
                #Mat.setValueStencil(row, col, val)
                self.Mat[row, col] = val
                # i+1
                if i < iMax-1 and self.d.vecArray[i+1,j] > 0:
                    #col.index = (i+1, j)
                    col = i + 1 * iMax * j
                    #Mat.setValueStencil(row, col, Ke)
                    self.Mat[row, col] = Ke
                # j+1
                if j < jMax-1 and self.d.vecArray[i,j+1] > 0:
                    #col.index = (i, j+1)
                    col = i + iMax * (j + 1)
                    #Mat.setValueStencil(row, col, Ks)    
                    self.Mat[row, col] = Ks    
        #Mat.assemble()
        self.Mat.assemblyBegin()
        self.Mat.assemblyEnd()
        return 0
    def set_Poisson_adj(self, ksp, MatPc, Mat):
        da = self.Phai.da
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        (i0, i1), (j0, j1) = da.getRanges()
        (iMax, jMax) = da.getSizes()
        #rows, cols, vals = [],[],[]
        #Istart, Iend = Mat.getOwnershipRange()
        ranges = da.getOwnershipRanges()
        #rankI = self.rank % ranges[0].size
        #rankJ = self.rank // ranges[0].size
        i, j = self.d.get_slice()

        d_c = self.d.vecArray[i[0],j[0]]
        d_w = self.d.vecArray[i[-1],j[0]]
        d_e = self.d.vecArray[i[1],j[0]] 
        d_n = self.d.vecArray[i[0],j[-1]]
        d_s = self.d.vecArray[i[0],j[1]] 
        dcw = (d_c + d_w) / 2
        dce = (d_c + d_e) / 2
        dcn = (d_c + d_n) / 2
        dcs = (d_c + d_s) / 2
        
        yM_array = np.ones(j1-j0)
        yN0array = np.ones(j1-j0)
        yN1array = np.ones(j1-j0)
        yM_array[:self.yM[j[0]].size ] = self.yM[j[0]]
        yN0array[:self.yM[j[0]].size ] = self.yN[j[0]]
        yN1array[:self.yM[j[0]].size ] = self.yN[j[1]]
        cos_c = np.cos(yM_array)
        cos_cn = np.cos(yN0array)
        cos_cs = np.cos(yN1array)
        cos_n = np.cos(self.yM[j[0]]+self.dy)
        cos_s = np.cos(self.yM[j[0]]-self.dy)
        K_w = d_c * dcw / (3 * (self.R * cos_c * self.dx)**2)
        K_e = d_c * dce / (3 * (self.R * cos_c * self.dx)**2)
        K_n = d_c * dcn / (3 * (self.R * self.dy)**2) * cos_cn / cos_c
        K_s = d_c * dcs / (3 * (self.R * self.dy)**2) * cos_cs / cos_c
        KT_w = d_w * dcw / (3 * (self.R * cos_c * self.dx)**2)
        KT_e = d_e * dce / (3 * (self.R * cos_c * self.dx)**2)
        KT_n = d_n * dcn / (3 * (self.R * self.dy)**2) * cos_cn / cos_n
        KT_s = d_s * dcs / (3 * (self.R * self.dy)**2) * cos_cs / cos_s
        for j in range(j0, j1):
            for i in range(i0, i1):
                row.index = (i, j)
                Kw = K_w[i-i0, j-j0] 
                Ke = K_e[i-i0, j-j0] 
                Kn = K_n[i-i0, j-j0] 
                Ks = K_s[i-i0, j-j0] 
                KTw = KT_w[i-i0, j-j0] 
                KTe = KT_e[i-i0, j-j0] 
                KTn = KT_n[i-i0, j-j0] 
                KTs = KT_s[i-i0, j-j0] 
                # j-1
                if j > 0 and self.d.vecArray[i,j-1] > 0:
                    col.index = (i, j-1)
                    Mat.setValueStencil(row, col, KTn)
                # i-1
                if i > 0 and self.d.vecArray[i-1,j]>0:
                    col.index = (i-1, j)
                    Mat.setValueStencil(row, col, KTw)
                # center
                val = -1 #- ((KTw + KTe) + (KTn + KTs))
                
                isFixBoundary = True
                if isFixBoundary:
                    if j>0      and self.d.vecArray[i,j-1]>0: val -= Kn
                    if i>0      and self.d.vecArray[i-1,j]>0: val -= Kw
                    if i<iMax-1 and self.d.vecArray[i+1,j]>0: val -= Ke
                    if j<jMax-1 and self.d.vecArray[i,j+1]>0: val -= Ks
                col.index = (i, j)
                Mat.setValueStencil(row, col, val)
                # i+1
                if i < iMax-1 and self.d.vecArray[i+1,j] > 0:
                    col.index = (i+1, j)
                    Mat.setValueStencil(row, col, KTe)
                # j+1
                if j < jMax-1 and self.d.vecArray[i,j+1] > 0:
                    col.index = (i, j+1)
                    Mat.setValueStencil(row, col, KTs)    
        Mat.assemble()
        return 0

    def forward(self, dt, with_Coriolis=True, is_reversal=False, with_sponge=True, with_Sommerfeld=True):
        self.copy_to_pre()
        if with_Sommerfeld:
            self.update_MN_by_Sommerfeld(dt)
        self.update_MN_by_h(dt)
        if with_Coriolis:
            self.update_MN_by_Coriolis(dt, is_reversal=is_reversal)
        if with_sponge:
            self.update_MN_by_sponge(dt)
        if self.has_Boussinesq:
            self.update_b_by_h()
            self.update_Phai_by_b()
            self.update_Phai_by_tapering()
            self.update_MN_by_Phai(dt)
        self.update_AMBN_by_MN(dt)
        if self.solid_earth is None:
            self.update_h_by_AMBN(dt, is_intermediate=False)
        else:
            self.update_h_by_AMBN_with_SAL(dt)
        self.update_hmax_by_h()
    def adjoint(self, dt, with_Coriolis=True, with_sponge=True, with_Sommerfeld=True):
        self.update_hmax_by_h()
        if self.solid_earth is None:
            self.update_h_by_AMBN_adj(dt, is_intermediate=False)
        else:
            self.update_h_by_AMBN_with_SAL_adj(dt)
        self.update_AMBN_by_MN_adj(dt)
        if self.has_Boussinesq:
            self.update_MN_by_Phai_adj(dt)
            self.update_Phai_by_tapering()
            self.update_Phai_by_b_adj()
            self.update_b_by_h_adj()
        if with_sponge:
            self.update_MN_by_sponge(dt)
        if with_Coriolis:
            self.update_MN_by_Coriolis_adj(dt)
        self.update_MN_by_h_adj(dt)
        if with_Sommerfeld:
            self.update_MN_by_Sommerfeld_adj(dt)
        self.copy_to_pre_adj()

    def add(self, variable, i, j, amount=1e-6):
        (i0, i1), (j0, j1) = variable.da.getRanges()
        if (i0 <= i < i1) and (j0 <= j < j1):
            variable.vecArray[i,j] += amount
        variable.local_to_local()

    def setup_okada(self, lonR, latR, strike, depth, AL, AW, dip, rake, displacement):
        (i0, i1), (j0, j1) = self.d.da.getRanges()
        lons = np.rad2deg(self.xN[i0:i1])
        lats = np.rad2deg(self.yM[j0:j1])
        #print("lonlat.shape: ", self.xN.shape, self.yM.shape, lons.shape, lats.shape, flush=True)
        self.okada = Okada(lons, lats)
        d_pad = self.d.vecArray[i0-1:i1+1,j0-1:j1+1].copy().T
        self.uz = self.okada.uz(lonR, latR, strike, depth, 
                                AL, AW, dip, rake,
                                displacement, d_pad).T
    def add_okada(self, displacement=1.0):
        (i0, i1), (j0, j1) = self.d.da.getRanges()
        self.h.vecArray[i0:i1,j0:j1] += self.uz * displacement
        self.h.local_to_local()

    def add_raised_cosine(self, variable, i, j, amplitude, radius):
        (i0, i1), (j0, j1) = variable.da.getRanges()
        x = self.R * (self.xN[i0:i1] - self.xN[i]) * np.cos(self.yM[j])
        y = self.R * (self.yM[j0:j1] - self.yM[j])
        #print(self.rank, "x", self.xN[i], self.xN[i0:i1])
        #print(self.rank, "y", self.yM[j], self.yM[j0:j1])
        xx, yy = np.meshgrid(y, x)
        #xx, yy = np.meshgrid(x, y, indexing='ij')
        #yy, xx = np.meshgrid(y, x)
        rr = np.sqrt(xx**2 + yy**2)
        variable.vecArray[i0:i1,j0:j1] += self.__raised_cosine(radius, rr) * amplitude
        variable.local_to_local()
    
    def add_raised_cosine_xy(self, variable, lon, lat, amplitude, radius):
        (i0, i1), (j0, j1) = variable.da.getRanges()
        x = self.R * (self.xN[i0:i1] - np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
        y = self.R * (self.yM[j0:j1] - np.deg2rad(lat))
        xx, yy = np.meshgrid(y, x)
        rr = np.sqrt(xx**2 + yy**2)
        variable.vecArray[i0:i1,j0:j1] += self.__raised_cosine(radius, rr) * amplitude
        variable.local_to_local()

    def add_numerical_delta(self, variable, i, j, amplitude):
        (i0, i1), (j0, j1) = variable.da.getRanges()
        if i0<=i<i1 and j0<=j<j1:
            variable.vecArray[i,j] += amplitude
        variable.local_to_local()

    def set_random(self, variable, min, max, seed=0):
        (i0, i1), (j0, j1) = variable.da.getRanges()
        np.random.seed(seed + self.rank)
        variable.vecArray[i0:i1,j0:j1] = min + (max - min) * np.random.rand(i1-i0, j1-j0)
        variable.local_to_local()

    def set_values_by_func(self, variable, func):
        (i0, i1), (j0, j1) = variable.da.getRanges()
        x, y = self.xN, self.yM
        if variable.location == "left":
            x = self.xM
        if variable.location == "down":
            y = self.yN
        xx, yy = np.meshgrid(x[i0:i1], y[j0:j1])
        variable.vecArray[i0:i1,j0:j1] = func(xx.T, yy.T)
        variable.local_to_local()

    def get_local_coods(self, variable):
        (i0, i1), (j0, j1) = variable.da.getRanges()
        xrad = self.xN
        yrad = self.yM
        if variable.location=="left":
            xrad = self.xM
        if variable.location=="down":
            yrad = self.yN
        lon_deg = np.rad2deg(xrad[i0:i1])
        lat_deg = np.rad2deg(yrad[j0:j1])
        da_lon = xr.DataArray(
            lon_deg, 
            coords=[lon_deg], 
            dims=["longitude"], 
            attrs={"units":"°",
                   "standard_name":"lon",
                   "long_name":"longitude"}, )
        da_lat = xr.DataArray(
            lat_deg, 
            coords=[lat_deg], 
            dims=["latitude"], 
            attrs={"units":"°",
                   "standard_name":"lat",
                   "long_name":"latitude"}, )
        return {"longitude":da_lon, "latitude":da_lat}

    def get_local_array(self, variable):
        (i0, i1), (j0, j1) = variable.da.getRanges()
        return np.array(variable.vecArray[i0:i1,j0:j1])

    def get_xr_data_array(self, variable):
        coods = self.get_local_coods(variable)
        array = self.get_local_array(variable)
        xda = xr.DataArray(array, coods, coods.keys())
        return xda
    
    def get_xr_data_array_recorder(self, variable, times, attrs=None):
        (i0, i1), (j0, j1) = variable.da.getRanges()
        da_time = xr.DataArray(
            times, 
            coords=[times], 
            dims=["time"], 
            attrs={"units":"sec",
                "standard_name":"time",})
        local_coods = self.get_local_coods(variable)
        coods = {"time": da_time}
        coods.update(local_coods)
        array = np.zeros((len(times), i1 - i0, j1 - j0))
        xda = xr.DataArray(array, coods, coods.keys(), attrs=attrs)
        return xda


    def get_max(self, variable):
        max_local = np.max(self.get_local_array(variable))
        max_rank0 = MPI.COMM_WORLD.gather(max_local, root=0)
        if self.rank==0:
            max_rank0 = np.max(max_rank0)
        else:
            max_rank0 = None
        max_global = MPI.COMM_WORLD.bcast(max_rank0, root=0)
        return max_global
    def get_min(self, variable):
        min_local = np.min(self.get_local_array(variable))
        min_rank0 = MPI.COMM_WORLD.gather(min_local, root=0)
        if self.rank==0:
            min_rank0 = np.min(min_rank0)
        else:
            min_rank0 = None
        min_global = MPI.COMM_WORLD.bcast(min_rank0, root=0)
        return min_global

    def get_filtered_h(self, ij_list):
        self.h_stencil.copy_from(self.h)
        values = []
        for i, j in ij_list:
            lat0 = self.yM[j]
            filt = self.get_filter_array(lat0, self.filter_radius)
            m = filt.shape[0] // 2
            n = filt.shape[1] // 2
            array = self.h_stencil.vecArray[i-m:i+m+1,j-n:j+n+1]
            values.append(np.sum(array * filt))
            #array[:] = signal.convolve2d(array, filt, mode="same")
            #values.append(array[m,n])
        return values
    def get_hMNUV(self, ij_list):
        #self.h_stencil.copy_from(self.h)
        _dh = self.d.get_values(ij_list)
        _dM = self.dM.get_values(ij_list)
        _dN = self.dN.get_values(ij_list)
        _h = self.h.get_values(ij_list)
        _M = self.M.get_values(ij_list)
        _N = self.N.get_values(ij_list)
        _U = _M / _dM
        _V = _N / _dN * -1 # convert to northward velocity
        values = np.array([_h,_M,_N,_U,_V]).T
        return values # (len(ij_list), 5)
    def get_filter_array(self, lat0, radius):
        dx = self.R * self.dx * np.cos(lat0)
        dy = self.R * self.dy
        nx = int(np.floor(radius / dx))
        ny = int(np.floor(radius / dy))
        x = np.arange(-nx, nx+1) * dx
        y = np.arange(-ny, ny+1) * dy
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx**2 + yy**2)
        filt = self.__raised_cosine(radius, rr)
        filt /= np.sum(filt)
        #print(f"dx:{dx}, dy:{dy}, nx:{nx}, ny:{ny}")
        #print(f"filter.radius: {radius}")
        #print(f"filter.shape: {filt.shape}" )
        return filt
    def __raised_cosine(self, r, x):
        a = np.abs(np.pi * x / r)
        return (1 + np.cos(a)) / 2 * (a < np.pi)


    def get_xr_dataset_h(self):
        names =["d", "h", "z", "h_pre", "z_pre", "b", "Phai", "AMBN", "h_star", "comp"]
        variables =[self.d, self.h, self.z, self.h_pre, 
                    self.z_pre, self.b, self.Phai, self.AMBN, self.h_star, self.comp]
        dataset ={}
        for name, variable in zip(names, variables):
            xda = self.get_xr_data_array(variable)
            dataset[name] = xda
        return xr.Dataset(dataset)
    def get_xr_dataset_M(self):
        names =["M", "M_pre"]
        variables =[self.M, self.M_pre]
        dataset ={}
        for name, variable in zip(names, variables):
            xda = self.get_xr_data_array(variable)
            dataset[name] = xda
        return xr.Dataset(dataset)
    def get_xr_dataset_N(self):
        names =["N", "N_pre"]
        variables =[self.M, self.M_pre]
        dataset ={}
        for name, variable in zip(names, variables):
            xda = self.get_xr_data_array(variable)
            dataset[name] = xda
        return xr.Dataset(dataset)


    def save_xr_dataset_in_parallel(self, file_name_header):
        # h
        xds = self.get_xr_dataset_h()
        path = f"{self.outpath}/{file_name_header}_h_{self.rank:04}.nc"
        if os.path.exists(path):
            os.remove(path)
        xds.to_netcdf(path)
        # M
        xds = self.get_xr_dataset_M()
        path = f"{self.outpath}/{file_name_header}_M_{self.rank:04}.nc"
        if os.path.exists(path):
            os.remove(path)
        xds.to_netcdf(path)
        # N
        xds = self.get_xr_dataset_N()
        path = f"{self.outpath}/{file_name_header}_N_{self.rank:04}.nc"
        if os.path.exists(path):
            os.remove(path)
        xds.to_netcdf(path)


if __name__=="__main__":
    shape = (5, 7)
    ocean = Ocean(shape, extent=(1,2,3,4))
    if ocean.rank==0:
        ocean.d.rank0[:] = np.arange(35).reshape(*shape)
    ocean.comm.Barrier()
    ocean.d.scatter()
    ocean.d.set_boundary(boundary_type="copy")
    ocean.d.print_vec_array("ocean.d")


