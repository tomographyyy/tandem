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

import numpy as np
from pyproj import Geod
class Okada(object):
    """Class for calculation of surface deformation in spherical coordinate system
    Okada, Y. (1985). Surface deformation due to shear and tensile faults in a half-space. 
        Bulletin of the Seismological Society of America, 75(4), 1135–1154. https://doi.org/10.1785/BSSA0750041135
    Okada, Y. (1992). Internal deformation due to shear and tensile faults in a half-space. 
        Bulletin of the Seismological Society of America, 82(2), 1018–1040. https://doi.org/10.1785/BSSA0820021018
    """
    def __init__(self, lons, lats, ellps="WGS84", Poisson=0.25):
        """init
        Args:
            lons (1d array): target longitudes [degree].
            lats (1d array): target latitudes  [degree].
            ellps (str, optional): ellipsoid name. Defaults to "WGS84".
            Poisson (float, optional): Poisson's ratio. Defaults to 0.25.
        """
        self.lons, self.lats = np.meshgrid(lons, lats)
        self.geod = Geod(ellps=ellps)
        self.Poisson = Poisson
    def okada_f(self, x, y, depth, AL, dip, AW):
        try:
            cos = np.cos(np.deg2rad(dip))
            sin = np.sin(np.deg2rad(dip))
            p = y * cos + depth * sin
            q = y * sin - depth * cos
            xi = x -AL
            eta = p - AW
            y_ = eta * cos + q * sin
            d_ = eta * sin - q * cos
            X = np.sqrt(xi**2+q**2)
            R = np.sqrt(X**2+eta**2)
            a = 1 - 2 * self.Poisson
            yy = eta*(X+q*cos)+X*(R+X)*sin
        except:
            import traceback
            print(np.min(y),np.max(y),depth , cos, sin)
            print(np.min(p),np.max(p),np.min(eta), np.max(eta) , cos, sin)
            traceback.print_exc()
        xx = xi*(R+X)*cos + 1e-30
        I5 = a * 2 / cos * np.arctan(yy / xx)
        I4 = a/cos*(np.log(R+d_)-sin*np.log(R+eta))
        I3=a*(y_/cos/(R+d_)-np.log(R+eta))+sin/cos*I4
        I2=a*(-np.log(R+eta)) - I3
        I1= a*(-1/cos*xi/(R+d_))-sin/cos*I5
        if type(xi)==np.ndarray:
            u_strike=np.empty((3,*xi.shape))
        else:
            u_strike=np.empty(3)
        u_strike[0] = xi*q/R/(R+eta)+np.arctan(xi*eta / (q*R)) + I1 * sin
        u_strike[1] = y_*q/R/(R+eta)+q*cos/(R+eta) + I2 * sin
        u_strike[2] = d_*q/R/(R+eta)+q*sin/(R+eta) + I4 * sin
        u_strike /= -2*np.pi
        u_dipslip = np.empty_like(u_strike)
        u_dipslip[0] = q / R - I3 * sin * cos
        u_dipslip[1] = y_*q / R / (R+xi) + cos * np.arctan(xi * eta / (q * R)) - I1 * sin * cos
        u_dipslip[2] = d_*q / R / (R+xi) + sin * np.arctan(xi * eta / (q * R)) - I5 * sin * cos
        u_dipslip /= -2 * np.pi
        return u_strike, u_dipslip
    def okada_local(self, x, y, depth, AL, dip, AW):
        AL=np.array(AL)
        AW=np.array(AW)
        f0 = self.okada_f(x, y, depth, AL[0], dip, AW[0])
        f1 = self.okada_f(x, y, depth, AL[1], dip, AW[0])
        f2 = self.okada_f(x, y, depth, AL[0], dip, AW[1])
        f3 = self.okada_f(x, y, depth, AL[1], dip, AW[1])
        u_strike = f0[0] - f1[0] - f2[0] + f3[0]
        u_dipslip = f0[1] - f1[1] - f2[1] + f3[1]
        return u_strike, u_dipslip
    def lonlat2xy(self, lonR, latR, strike):
        lonR = np.ones_like(self.lons) * lonR
        latR = np.ones_like(self.lons) * latR
        az, _, distance = self.geod.inv(lonR, latR, self.lons, self.lats, radians=False)
        x = distance * np.sin(-np.deg2rad(90-(az-(strike-180))))
        y = distance * np.cos(-np.deg2rad(90-(az-(strike-180))))
        return x, y
    def __u_global(self, lonR, latR, strike, depth, AL, AW,
                            dip, rake, displacement):
        x, y = self.lonlat2xy(lonR, latR, strike)
        u1_strike, u1_dipslip = self.okada_local(x, y, depth, AL, dip, AW)
        ux, uy, uz = displacement * (u1_strike * np.cos(np.deg2rad(rake)) 
                                + u1_dipslip * np.sin(np.deg2rad(rake)))
        uE = ux * np.sin(np.deg2rad(strike)) - uy * np.cos(np.deg2rad(strike)) # corrected
        uN = ux * np.cos(np.deg2rad(strike)) + uy * np.sin(np.deg2rad(strike)) # corrected
        return uE, uN, uz
    def u_global(self, lonR, latR, strike, depth, AL, AW,
                            dip, rake, displacement):
        if type(lonR) is list:
            uE = np.zeros_like(self.lats)
            uN = uE.copy()
            uz = uE.copy()
            for i in range(len(lonR)):
                _uE, _uN, _uz = self.__u_global(lonR[i], latR[i], strike[i], 
                                                depth[i], AL[i], AW[i],
                                                dip[i], rake[i], displacement[i])
                uE += _uE
                uN += _uN
                uz += _uz
            return uE, uN, uz
        else:
            return self.__u_global(lonR, latR, strike, depth, AL, AW,
                            dip, rake, displacement)
    def uz_horizontal(self, h_pad, uE, uN, uz):
        dlon = self.lons[0,1] - self.lons[0,0]
        dlat = self.lats[1,0] - self.lats[0,0]
        lat0 = np.mean(self.lats[:,0])
        dE = self.geod.a * np.deg2rad(dlon) * np.cos(np.deg2rad(lat0))
        dN = self.geod.b * np.deg2rad(dlat)
        dhdE = (h_pad[1:-1,2:] - h_pad[1:-1,:-2]) / (2 * dE)
        dhdN = (h_pad[2:,1:-1] - h_pad[:-2,1:-1]) / (2 * dN)
        return uE * dhdE + uN * dhdN
    def uz(self, lonR, latR, strike, depth, AL, AW, dip, rake,
            displacement, h_pad):
        uE, uN, uz = self.u_global(lonR, latR, strike, depth, AL, AW, 
                                dip, rake, displacement)
        uzh = self.uz_horizontal(h_pad, uE, uN, uz)
        return uz + uzh