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
            csvfile = "Greens function with SAL effects with sign correction.csv"
            df_GreensFunc = pandas.read_csv(csvfile, comment='#')
            thetaRad=np.array(list(df_GreensFunc["theta[deg]"]))*(np.pi/180)
            thetaRad[0]=0.0 # set the initial data point to zero
            uG     = np.array(list(df_GreensFunc["uG_Pagiatakis[m]"]))
            uG_SAL = np.array(list(df_GreensFunc["uG_SAL[m]"]))
            uG_Itp     = InterpolatedUnivariateSpline(thetaRad,uG,    k=1)
            uG_SAL_Itp = InterpolatedUnivariateSpline(thetaRad,uG_SAL,k=1)

            lmax = sh_n - 1   # maximum degree of spherical harmonic representation.
            mmax = sh_n - 1   # maximum order of spherical harmonic representation.
            #print ("lmax, mmax", lmax, mmax, "5555555555555555555555555555555", flush=True)
            #print(self.ocean.xM[0:3])
            
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
            #self.spatGreenNew = self.sh.synth(self.specGreen_lm) 
            tstart = time.time()
            #m=0成分のコピー
            if False:
                for i in range(self.specGreen_lm.size):    # this is a major bottleneck (~50sec)
                    if (self.sh.m[i] > 0):
                        self.specGreen_lm[i]=self.specGreen_lm[self.sh.l[i]]
            else:
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
        # np.savez("sal_new", spatLoad=spatLoad, specLoad=specLoad, specNew=specNew, spatNew=spatNew, z_coarse=z_coarse, shl = self.sh.l)

        return z_coarse

    def to_local(self, spatNew):
        return spatNew[self.j0:self.j1,self.i0:self.i1]
    def to_global(self, h_star_coarse):
        jend = min(self.j1, self.j0 + h_star_coarse.shape[0])
        iend = min(self.i1, self.i0 + h_star_coarse.shape[1])
        if False:
            print("################## for debug ###########################")
            print("self.j0:", self.j0)
            print("self.j1:", self.j1)
            print("jend:", jend)
            print("self.i0:", self.i0)
            print("self.i1:", self.i1)
            print("iend:", iend)
            print("h_star_coarse.shape:", h_star_coarse.shape)
            print("################## for debug ###########################")
        self.h[self.j0:jend,self.i0:iend] = h_star_coarse

class SolidEarth_old:
    def __init__(self, geoSphere, GID, sh_n, IGF):
        import pandas
        csvfile = "Greens function with SAL effects with sign correction.csv"
        self.use_shtns = True
        #if self.use_shtns:
        #    import shtns
        from scipy.interpolate import InterpolatedUnivariateSpline
        #print ("line 660", flush=True)
        self.geoSphere = geoSphere
        #print ("line 662", flush=True)
        df_GreensFunc = pandas.read_csv(csvfile, comment='#')
        thetaRad=np.array(list(df_GreensFunc["theta[deg]"]))*(np.pi/180)
        thetaRad[0]=0.0 #最小のデータ点をゼロに設定
        uG     = np.array(list(df_GreensFunc["uG_Pagiatakis[m]"]))
        uG_SAL = np.array(list(df_GreensFunc["uG_SAL[m]"]))
        uG_Itp     = InterpolatedUnivariateSpline(thetaRad,uG,    k=1)
        uG_SAL_Itp = InterpolatedUnivariateSpline(thetaRad,uG_SAL,k=1)



        lmax = sh_n - 1   # maximum degree of spherical harmonic representation.
        mmax = sh_n - 1   # maximum order of spherical harmonic representation.
        print ("lmax, mmax", lmax, mmax, "5555555555555555555555555555555", flush=True)
        print(self.geoSphere.xM[0:3])
        
        self.sh = shtns.sht(lmax, mmax)     # create sht object with given lmax and mmax (orthonormalized)
        nlat, nphi = self.sh.set_grid()     # build default grid (gauss grid, phi-contiguous)
        #print(self.sh.nlat, self.sh.nphi,sh_n, flush=True)   # displays the latitudinal and longitudinal grid sizes.
        """
        if self.sh.nlat != sh_n:
            mes = "sh.nlat != sh_n; sh.nlat={}, sh_n={}".format(self.sh.nlat, sh_n) \
                + "\n" + "sh_n must be multiples of 4."
            print ("ValueError: " + mes, flush=True)
            raise ValueError(mes)
        if self.sh.nphi != 2 * sh_n:
            mes = "sh.nphi != 2 * sh_n; sh.nphi={}, sh_n={}".format(self.sh.nphi, sh_n) \
                + "\n" + "sh_n must be multiples of 4 and the number of synthesis with small prime numbers {2,3,5,7}."
            print ("ValueError: " + mes, flush=True)
            raise ValueError(mes)
        """
        theta = np.arccos(self.sh.cos_theta)[:]
        #print ("line 712", flush=True)
        #print ("sh.cos_theta.shape: ", self.sh.cos_theta.shape, flush=True)

        #print("IGF:", IGF, flush=True)
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

        print ("line 724", flush=True)
        #print ("uG_New.shape: ", uG_New.shape, flush=True)
        #print ("sh.spat_shape: ", self.sh.spat_shape, flush=True)
        spatGreen = np.dot(np.diag(uG_New), np.ones(self.sh.spat_shape)) 
        #グリーン関数のスペクトル係数を求める
        #print ("line 727", flush=True)
        self.specGreen_lm = self.sh.analys(spatGreen)
        #print ("line 729", flush=True)
        self.spatGreenNew = self.sh.synth(self.specGreen_lm) 
        time_764 = time.time()
        #m=0成分のコピー
        for i in range(self.specGreen_lm.size):    #ここのfor文がボトルネック(~50sec)
            if (self.sh.m[i] > 0):
                self.specGreen_lm[i]=self.specGreen_lm[self.sh.l[i]]
        print ("line 769", time.time() - time_764, flush=True)
        if np.max(geoSphere.xM) < np.pi:
            xM = np.linspace(-1.0 * np.pi,  1.0 * np.pi, self.sh.nphi + 1, endpoint = True)  # data is alined from West  to East  [rad]
        else:
            xM = np.linspace(0,  2 * np.pi, self.sh.nphi + 1, endpoint = True)  # data is alined from West  to East  [rad]
        yN = np.linspace( 0.5 * np.pi, -0.5 * np.pi, self.sh.nlat + 1, endpoint = True)  # data is alined from North  to South  [rad]
        #print ("line 738", flush=True)
        #xM_Global = np.linspace(-1 * np.pi, 1 * np.pi, sh.nphi + 1, endpoint = True)  # data is alined from West  to East  [rad]
        self.xN = 0.5 * (xM[1:] + xM[:-1])
        self.yM = 0.5 * (yN[1:] + yN[:-1])#0.5 * np.pi - np.arccos(self.sh.cos_theta) 
        #yN = np.hstack((0.5 * np.pi, 0.5 * (self.yM[1:] + self.yM[:-1]), -0.5 * np.pi))
        self.h = np.zeros((self.sh.nlat,self.sh.nphi),dtype=float)
        gsxM = geoSphere.xM.copy()
        if gsxM[0]<0:
            gsxM += 2 * np.pi
        #print("np.min(gsxM):", np.min(gsxM))
        self.iMin = np.max(np.where(xM < np.min(gsxM)))
        self.iMax = np.min(np.where(xM > np.max(gsxM)))
        self.jMin = np.max(np.where(yN > np.max(geoSphere.yN)))
        self.jMax = np.min(np.where(yN < np.min(geoSphere.yN)))
        #print("line799 iMin, iMax, jMin, jMax:", self.iMin, self.iMax, self.jMin, self.jMax)
        xo = geoSphere.xM
        yo = geoSphere.yN
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
                
    #def to_local_old(self, spatNew):
    #    g2l = scipy.interpolate.interp2d(self.xN[self.iMin:self.iMax],self.yM[self.jMin:self.jMax],spatNew[self.jMin:self.jMax,self.iMin:self.iMax])
    #    return g2l(self.geoSphere.xN,self.geoSphere.yM).ravel()     
    def to_local_same_mesh(self, spatNew):
        return spatNew[self.j0:self.j1,self.i0:self.i1].ravel()     
    def to_local_refine_mesh(self, spatNew):
        i0, i1, j0, j1, rf = self.i0, self.i1, self.j0, self.j1, self.rf
        mo = (i1 - i0) * rf
        no = (j1 - j0) * rf
        zs = spatNew.reshape(self.sh.nlat, self.sh.nphi)
        zo = np.zeros((no, mo))
        #print(i0,i1,j0,j1,no,mo,zo.shape,rf)
        for j in range(rf):
            dj = j // ((rf + 1) // 2) - 1
            for i in range(rf):
                di = i // ((rf + 1) // 2) - 1
                z00 = zs[j0+dj  :j1+dj  , i0+di  :i1+di  ]
                z01 = zs[j0+dj  :j1+dj  , i0+di+1:i1+di+1]
                z10 = zs[j0+dj+1:j1+dj+1, i0+di  :i1+di]
                z11 = zs[j0+dj+1:j1+dj+1, i0+di+1:i1+di+1]
                ri = (rf - 1 - 2 * i) % (2 * rf) // (2. * rf)
                rj = (rf - 1 - 2 * j) % (2 * rf) // (2. * rf)
                a00 =    rj    *    ri 
                a01 =    rj    * (1 - ri)
                a10 = (1 - rj) *    ri
                a11 = (1 - rj) * (1 - ri)
                #print(zo.shape, a00.shape, z00.shape)
                zo[j::rf,i::rf] = a00 * z00 + a01 * z01 + a10 * z10 + a11 * z11
        return zo.ravel()

    def to_local(self, spatNew):
        '''
        from scipy.interpolate import RegularGridInterpolator
        x_l = self.geoSphere.xN_coarse[:-1]
        y_l = self.geoSphere.yM_coarse[:-1] * -1 # to be ascending 
        x_g = self.xN
        y_g = self.yM * -1 # to be ascending 
        ig0 = np.max(np.where(x_g < np.min(x_l)))
        ig1 = np.min(np.where(x_g > np.max(x_l)))
        jg0 = np.max(np.where(y_g < np.min(y_l)))
        jg1 = np.min(np.where(y_g > np.max(y_l)))
        x_g_include_l = x_g[ig0:ig1+1]
        y_g_include_l = y_g[jg0:jg1+1]
        z_mg_dl = z_global[jg0:jg1+1, ig0:ig1+1] # mg: global mesh; dl: local domain
        xx, yy = np.meshgrid(x_l, y_l)
        g2l = RegularGridInterpolator((y_g_include_l, x_g_include_l), z_mg_dl)
        pts = np.array([yy.ravel(), xx.ravel()]).transpose()
        z_ml_dl = g2l(pts)
        return z_ml_dl.reshape(len(y_l),len(x_l))
        '''
        return spatNew[self.j0:self.j1,self.i0:self.i1]
    def to_global_old(self,hStar):
        l2g = scipy.interpolate.interp2d(self.geoSphere.xN,self.geoSphere.yM,hStar)
        self.h[self.jMin:self.jMax,self.iMin:self.iMax] = l2g(self.xN[self.iMin:self.iMax],self.yM[self.jMin:self.jMax])
    def to_global_same_mesh(self,hStar):
        self.h[self.j0:self.j1,self.i0:self.i1] = hStar
    def to_global_refine_mesh(self, hStar):
        hgl = self.h[self.j0:self.j1,self.i0:self.i1]
        hgl[:,:]=0
        for j in range(self.rf):
            for i in range(self.rf):
                hgl[:,:] += hStar[j::self.rf,i::self.rf]
        hgl /= self.rf**2

    def to_global(self, h_star_coarse):
        jend = min(self.j1, self.j0 + h_star_coarse.shape[0])
        iend = min(self.i1, self.i0 + h_star_coarse.shape[1])
        #print ("js: ", self.j0, self.j1, jend, flush=True)
        #print ("is: ", self.i0, self.i1, iend, flush=True)
        #print ("h_star_coarse.shape: ", h_star_coarse.shape, flush=True)
        self.h[self.j0:jend,self.i0:iend] = h_star_coarse
