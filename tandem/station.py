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

import pandas
import obspy
import numpy as np
from scipy import signal

class Angle(object):
    def __init__(self):
        pass
    def get_unit_vec(self, angle):
        return np.array([np.cos(angle), np.sin(angle)])
    def get_nearest_index(self, x, x_arr, extent, mid):
        if True: # ignore __is_in()
            dx = np.cos(x_arr) - np.cos(x)
            dy = np.sin(x_arr) - np.sin(x)
            r = dx**2 + dy**2
            idx = np.argmin(r)
            #print("idx", idx)
            return idx
        else:
            if self.__is_in(x, extent, mid):
                dx = np.cos(x_arr) - np.cos(x)
                dy = np.sin(x_arr) - np.sin(x)
                r = dx**2 + dy**2
                return np.argmin(r)
            else:
                return None
    def get_nearest_indexes(self, x_dst, x_src, extent, mid):
        result =[]
        for x in x_dst:
            idx = self.get_nearest_index(x, x_src, extent, mid)
            result.append(idx)
            if idx is None:
                print("=== x        =========================================")
                print(x)
                print("=== x_src[0] =========================================")
                print(x_src[0])
                print("=== x_src[-1] ========================================")
                print(x_src[-1])
                print("=== extent    ========================================")
                print(extent)
                print("=== mid       ========================================")
                print(mid)
                print("######################################################")
                break
        return result

    def __is_in(self, x, extent, mid):
        rx = self.get_unit_vec(x)
        r0 = self.get_unit_vec(extent[0])
        r1 = self.get_unit_vec(extent[1])
        rm = self.get_unit_vec(mid)
        # print(x, extent, mid, np.cross(r0, rm), flush=True)
        is_in0 = (np.cross(r0, rm) * np.cross(r0, rx)) >= 0
        is_in1 = (np.cross(r1, rm) * np.cross(r1, rx)) >= 0
        return is_in0 and is_in1
    
class Station(object):
    def __init__(self, csv_file):
        self.dataframe = pandas.read_csv(csv_file, comment="#")
    
    def is_near_source(self, xh, yh, xextent, yextent, wx, wy):
        x0 = self.dataframe.lon[0] % 360
        y0 = self.dataframe.lat[0]
        xextent=np.array(xextent)
        yextent=np.array(yextent)
        flag = True
        flag *= xextent.max() > x0 - wx
        flag *= xextent.min() < x0 + wx
        flag *= yextent.max() > y0 - wy
        flag *= yextent.min() < y0 + wy
        return flag

    def set_nearest_index(self, xh, yh, xextent, yextent, xmid, ymid, iextent, jextent):
        self.dataframe["i"] = ""
        self.dataframe["j"] = ""
        angle = Angle()
        drops =[]
        for k in range(len(self.dataframe)):
            x = np.deg2rad(self.dataframe.lon[k])
            y = np.deg2rad(self.dataframe.lat[k])
            i = angle.get_nearest_index(x, xh, xextent, xmid)
            j = angle.get_nearest_index(y, yh, yextent, ymid)
            if (i is not None) and (j is not None) \
                and iextent[0] <= i < iextent[1] \
                and jextent[0] <= j < jextent[1]:
                self.dataframe.at[k, "i"] = i
                self.dataframe.at[k, "j"] = j
            else:
                drops.append(k)
        #print("drops", drops)
        self.dataframe = self.dataframe.drop(self.dataframe.index[drops])
        #print("len(df)", len(self.dataframe))

    def get_list_stats(self):
        stats_list = []
        for _, row in self.dataframe.iterrows():
            stats_list.append({
                "network": row.network,
                "station": str(row.station),
                "location": row.location[:11],
                "channel": str(row.channel),
                "starttime": self.starttime,
                "delta": self.delta,
            })
        return stats_list

    def get_list_ij(self):
        ij_list = []
        for _, row in self.dataframe.iterrows():
            ij_list.append((row.i, row.j))
        return ij_list
    
    def save_csv(self, filename):
        if len(self.dataframe) > 0:
            #print(self.dataframe.columns)
            #self.dataframe.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
            self.dataframe.to_csv(filename)

    def setup_logger(self, nstep, delta, starttime="2000-01-01T00:00:00.000000Z"):
        self.logger = np.zeros((len(self.dataframe), 5, nstep))
        self.delta = delta
        self.starttime = starttime

    def record(self, step, values):
        if values.shape[0]>0:
            self.logger[:, 5, step] = values # (nstation, nvalues, nstep)
    
    def save_timeseries(self, filename):
        if len(self.dataframe) > 0:
            headers = self.get_list_stats()
            #traces=[]
            for k in range(len(self.dataframe)):
                stid = int(headers[k]["station"])
                data = {
                    "h": self.logger[k,0,:],
                    "M": self.logger[k,1,:],
                    "N": self.logger[k,2,:],
                    "U": self.logger[k,3,:],
                    "V": self.logger[k,4,:],
                }
                np.savez(filename[:-8] + f"_station{stid:04}.npz", **data)
                #traces.append(obspy.Trace(self.logger[k], headers[k]))
            #stream = obspy.Stream(traces)
            #stream.write(filename, format="MSEED")

class RaisedCosineFilter(object):
    def __init__(self, lat0, r, dlon, dlat, R = 6378137):
        self.lat0 = lat0
        self.r = r
        self.R = R
        self.stencil_width, self.filter_array = self.get_filter_array(dlon, dlat)
    def __raised_cosine(self, r, x):
        a = np.abs(np.pi * x / r)
        return (1 + np.cos(a)) / 2 * (a < np.pi)
    def get_filter_array(self, dlon, dlat):
        dx = self.R * dlon * np.cos(self.lat0)
        dy = self.R * dlat
        nx = int(np.floor(self.r / dx))
        ny = int(np.floor(self.r / dy))
        stencil_width = np.maximum(nx, ny)
        x = np.arange(-nx, nx+1) * dx
        y = np.arange(-ny, ny+1) * dy
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx**2 + yy**2)
        filt = self.__raised_cosine(self.r, rr)
        filt /= np.sum(filt)
        return stencil_width, filt
    def apply_filter(self, array):
        array[:] = signal.convolve2d(array, self.filter_array, mode="same")