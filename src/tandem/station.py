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
        self.dataframe = pandas.read_csv(csv_file)

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
                "channel": row.channel,
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

    def setup_logger(self, num, delta, starttime="2000-01-01T00:00:00.000000Z"):
        self.logger = np.zeros((len(self.dataframe), num))
        self.delta = delta
        self.starttime = starttime

    def record(self, step, values):
        self.logger[:,step] = values

    def save_timeseries(self, filename):
        if len(self.dataframe) > 0:
            headers = self.get_list_stats()
            traces=[]
            for k in range(len(self.dataframe)):
                stid = int(headers[k]["station"])
                np.save(filename[:-10] + f"_station{stid:04}.npy", self.logger[k])
                traces.append(obspy.Trace(self.logger[k], headers[k]))
            stream = obspy.Stream(traces)
            stream.write(filename, format="MSEED")

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