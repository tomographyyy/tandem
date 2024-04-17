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

import shtns
import pandas
import numpy as np
import time
from scipy.interpolate import InterpolatedUnivariateSpline
from mpi4py import MPI
import importlib

class SolidEarth:

    def __init__(self, ocean, GID, sh_n, IGF):
        if MPI.COMM_WORLD.rank==0:
            self.conj = True
            self.ocean = ocean
            csvfile = "data/Greens_function_of_SAL_effects.csv"
            df_GreensFunc = pandas.read_csv(csvfile, comment='#')
            thetaRad=np.array(list(df_GreensFunc["theta[deg]"]))*(np.pi/180)
            thetaRad[0]=0.0 # set the initial data point to zero
            uG     = np.array(list(df_GreensFunc["uG_Pagiatakis[m]"]))
            uG_SAL = np.array(list(df_GreensFunc["uG_SAL[m]"]))
            uG_Itp     = InterpolatedUnivariateSpline(thetaRad,uG,    k=1)
            uG_SAL_Itp = InterpolatedUnivariateSpline(thetaRad,uG_SAL,k=1)

            lmax = sh_n - 1   # maximum degree of spherical harmonic representation.
            mmax = sh_n - 1   # maximum order of spherical harmonic representation.
            
            self.sh = shtns.sht(lmax, mmax)     # create sht object with given lmax and mmax (orthonormalized)
            nlat, nphi = self.sh.set_grid()     # build default grid (gauss grid, phi-contiguous)
            theta = np.arccos(self.sh.cos_theta)[:]

            if IGF: # use integrated Green's function
                n_large = 2**10 * self.sh.nlat
                yN = np.linspace(0, np.pi, num = n_large + 1)
                yM = (yN[1:]+yN[:-1]) / 2
                if (GID):
                    uG_fine = uG_SAL_Itp(yM)
                else:
                    uG_fine = uG_Itp(yM)
                
                w = n_large // self.sh.nlat
                uG_New = np.zeros(self.sh.nlat)
                for i in range(self.sh.nlat):
                    r2sum = 0
                    zsum = 0
                    for k in range(w):
                        y=yM[i*w+k]
                        r2sum += y**2
                        zsum += uG_fine[i*w+k] * y**2
                    uG_New[i] = zsum / r2sum
            else:
                if(GID):
                    uG_New = uG_SAL_Itp(theta)
                else:
                    uG_New = uG_Itp(theta)

            spatGreen = np.dot(np.diag(uG_New), np.ones(self.sh.spat_shape)) 
            # set spectral coefficients
            self.specGreen_lm = self.sh.analys(spatGreen)
            tstart = time.time()
            # copy of components of m=0
            self.specGreen_lm[self.sh.m > 0] = \
                [self.specGreen_lm[self.sh.l[i]] \
                    for i in range(self.specGreen_lm.size) \
                        if (self.sh.m[i] > 0)]
            print (f"time to initialize specGreen_lm: {time.time() - tstart} sec", flush=True)
            if np.max(self.ocean.xM) < np.pi:
                xM = np.linspace(-1.0 * np.pi,  1.0 * np.pi, self.sh.nphi + 1, endpoint = True)  # data is alined from West  to East  [rad]
            else:
                xM = np.linspace(0,  2 * np.pi, self.sh.nphi + 1, endpoint = True)  # data is alined from West  to East  [rad]
            yN = np.linspace( 0.5 * np.pi, -0.5 * np.pi, self.sh.nlat + 1, endpoint = True)  # data is alined from North  to South  [rad]
            self.xN = 0.5 * (xM[1:] + xM[:-1])
            self.yM = 0.5 * (yN[1:] + yN[:-1]) 
            self.h = np.zeros((self.sh.nlat,self.sh.nphi),dtype=float)
            gsxM = self.ocean.xM.copy()
            if gsxM[0]<0:
                gsxM += 2 * np.pi
            self.iMin = np.max(np.where(xM < np.min(gsxM)))
            self.iMax = np.min(np.where(xM > np.max(gsxM)))
            self.jMin = np.max(np.where(yN > np.max(self.ocean.yN)))
            self.jMax = np.min(np.where(yN < np.min(self.ocean.yN)))
            xo = self.ocean.xM
            yo = self.ocean.yN
            xs = xM
            ys = yN
            rf = (xs[1]-xs[0]) / (xo[1]- xo[0])
            dxs = xs[1] - xs[0]
            dxo = xo[1] - xo[0]
            self.rf = dxs / dxo
            if np.abs(self.rf - round(rf)) > 0.001:
                raise ValueError("ratio of solid earth's dx to ocean's must be integer. the ratio is {}".format(self.rf))
            self.rf = int(round(rf))
            self.i0 = np.argmin(np.abs(xs-xo[0])) # i_min on solid earth mesh
            self.i1 = np.argmin(np.abs(xs-xo[-1]))
            self.j0 = np.argmin(np.abs(ys-yo[0]))
            self.j1 = np.argmin(np.abs(ys-yo[-1]))

    def get_z_coarse_by_SAL(self, h_star_coarse, nlevel):
        self.to_global(h_star_coarse)
        spatLoad = self.h * (1030 * self.ocean.R**2) # convert water surface elevation to load
        specLoad = self.sh.analys(spatLoad)
        if self.conj:
            specNew = specLoad * self.specGreen_lm.conjugate() \
                * np.sqrt(4.0 * np.pi / (2.0 * self.sh.l + 1.0))
        else:
            specNew = specLoad * self.specGreen_lm \
                * np.sqrt(4.0 * np.pi / (2.0 * self.sh.l + 1.0))
        spatNew = self.sh.synth(specNew)
        z_coarse = self.to_local(spatNew)
        
        return z_coarse

    def to_local(self, spatNew):
        return spatNew[self.j0:self.j1,self.i0:self.i1]
    def to_global(self, h_star_coarse):
        jend = min(self.j1, self.j0 + h_star_coarse.shape[0])
        iend = min(self.i1, self.i0 + h_star_coarse.shape[1])
        self.h[self.j0:jend,self.i0:iend] = h_star_coarse
