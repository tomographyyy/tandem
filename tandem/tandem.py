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

from email import header
import sys
import os
import numpy as np
from mpi4py import MPI
from tandem.ocean import Ocean
from tandem.solidearth import SolidEarth
from tandem.okada import Okada
from mpi4py import MPI
import xarray as xr
from tandem.decompi import DecoratorMPI
import cProfile
decompi = DecoratorMPI()

class Tandem(object):
    @decompi.finalize
    def __init__(self, outpath="", job_id="0000000", job_name="tandem"):
        print("outpath:", outpath)
        print("job_id:", job_id)
        print("job_name:", job_name, flush=True)
        self.outpath = outpath
        self.job_id = job_id
        self.Nonlinear = "_NL" in job_name
        self.remove_land = False
        if "_rmland" in job_name:
            self.remove_land = float(job_name.split("_rmland")[1][:4])
        print("Nonlinear:", self.Nonlinear, flush=True)
        self.job_name, job_option, source_option, manning_option = job_name.split("_")[:4]
        print("manning_option:", manning_option, flush=True)
        self.CMP = "C1" in job_option
        self.SAL = "S1" in job_option
        self.BSQ = "B1" in job_option
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        if "R" in job_option:
            self.resolution = int(job_option.split("R")[1])
        else:
            self.resolution = 120 #120 # 240 # arcsec
        Manning = float(manning_option[7:])
        #self.RSL = self.resolution // 30
        #self.RLV = int(np.round(np.log2(self.RSL)))
        area_type = "all"#"all"
        if area_type == "small":
            #self.NX = 28 * 60 * 60 // self.resolution #180 * 8 // self.RSL#30*4
            #self.NY = 36 * 60 * 60 // self.resolution #8 // self.RSL#30*4 
            #self.shape = (self.NX, self.NY)
            #self.extent = np.deg2rad([180+16*2, 240, 64, 12+16]) ### [xmin, xmax, ymax, ymin]
            self.NX = 8 * 60 * 60 // self.resolution #180 * 8 // self.RSL#30*4
            self.NY = 8 * 60 * 60 // self.resolution #8 // self.RSL#30*4 
            self.shape = (self.NX, self.NY)
            self.extent = np.deg2rad([224, 232, 56, 48]) ### [xmin, xmax, ymax, ymin]
        elif area_type == "north":
            _range = [188, 240, 64, 36]
            self.NX = (_range[1]-_range[0]) * 60 * 60 // self.resolution
            self.NY = (_range[2]-_range[3]) * 60 * 60 // self.resolution
            self.shape = (self.NX, self.NY)
            self.extent = np.deg2rad(_range) ### [xmin, xmax, ymax, ymin]
        elif area_type == "northeast":
            _range = [192, 240, 64, 32]
            self.NX = (_range[1]-_range[0]) * 60 * 60 // self.resolution
            self.NY = (_range[2]-_range[3]) * 60 * 60 // self.resolution
            self.shape = (self.NX, self.NY)
            self.extent = np.deg2rad(_range) ### [xmin, xmax, ymax, ymin]
        else:
            self.NX = 76 * 60 * 60 // self.resolution #180 * 8 // self.RSL#30*4
            self.NY = 68 * 60 * 60 // self.resolution #8 // self.RSL#30*4 
            self.shape = (self.NX, self.NY)
            self.extent = np.deg2rad([180-16, 240, 64, 12-16]) ### [xmin, xmax, ymax, ymin]
        if self.SAL:
            if self.resolution==120:
                self.CLV = 3
            elif self.resolution==60:
                self.CLV = 4
            elif self.resolution==30:
                self.CLV = 5
            #self.CLV = 4 #3 #5 - self.RLV #0 #5 # 1: shn=1350, 2: shn=675
        else:
            self.CLV = 0
        if self.job_name == "forward":
            filter_radius = 1#200e3 #500e3 #5e3
        else:
            filter_radius = 1#200e3 # 1  #500e3#1
        self.ocean = Ocean(self.shape, extent=self.extent, damping_factor=0.1*1,
                            CLV=self.CLV, has_Boussinesq=self.BSQ,
                            outpath=self.outpath, filter_radius=filter_radius, 
                            Manning=Manning, Nonlinear=self.Nonlinear)         # $$$$$$$$$$$$$$$$$$$$ Manning (0.025)$$$$$$$$$$$$$$$$$$$$$$$$$$$
        filename = "gebco/GEBCO_2019_030sec_minimum_NorthPacific_correction_Canada_LA_Oahu_*.nc"
        #filename = "OpenTsunami15sec_chunks/WholeJapan2020*_*.nc"
        self.ocean.load_bathymetry(filename, depth=False, lon="lon", lat="lat", z="elevation")
        #self.ocean.set_random(variable=self.ocean.d, min=-0.05, max=0.95, seed=1)
        #self.ocean.set_random(variable=self.ocean.d, min=1.0, max=1.0, seed=1)
        #xmin = np.min(self.ocean.xN)
        #xmax = np.max(self.ocean.xN)
        #ymin = np.min(self.ocean.yM)
        #ymax = np.max(self.ocean.yN)
        #ax = 4000 / (xmax - xmin) * 1
        #ay = 4000 / (ymax - ymin) * 1
        #xmid = np.deg2rad(180)
        #ymid = np.deg2rad(40)
        #####################
        # depth
        #####################
        if False: # depth defined by function
            self.ocean.set_values_by_func(variable=self.ocean.d, 
                #func = lambda x, y: 1) # uniform
                #func = lambda x, y: (1 + x / 0.14 ) / 2) # linear x
                #func = lambda x, y: 0.01 + 0.99 * (x + 0.07) / 0.14   # linear x 100
                #func = lambda x, y: (1 + (x + y) / 0.14 ) / 2 # linear xy
                # func = lambda x, y: 10 + ax * (x - xmin) + ay * (y - ymin)  # linear xy
                #func = lambda x, y: 3990 * ((y-ymid) < (x-xmid) / 64) + 200  # shelf
                func = lambda x, y: 4000  # uniform
                #func = lambda x, y: (1 + (x + y > 0)) / 2 # step xy
                #func = lambda x, y: (1 + (x > 0)) / 2 # step x
                #func = lambda x, y: 0.01 * (x < 0) + 0.99 * (x >= 0) # step x 100
                )#"""
        if False: # depth with a line reflector
            (i0, i1), (j0, j1) = self.ocean.d.da.getRanges()
            self.ocean.d.vecArray[i0:i1,j0:j1] = 4000
            i_slc = slice(np.maximum(0, i0), np.minimum(self.NX//2, i1))
            j_slc = slice(np.maximum(self.NY//2, j0), np.minimum(self.NY//2 + 1, j1))
            self.ocean.d.vecArray[i_slc, j_slc] = -100
            self.ocean.d.local_to_local()
        if False: # depth with parallel reflectors
            (i0, i1), (j0, j1) = self.ocean.d.da.getRanges()
            self.ocean.d.vecArray[i0:i1,j0:j1] = 4000
            if i0<=120<i1:
                self.ocean.d.vecArray[120,j0:j1] = -100
            if i0<=240<i1:
                self.ocean.d.vecArray[240,j0:j1] = -100
            self.ocean.d.local_to_local()
        if False: # double atolls
            (i0, i1), (j0, j1) = self.ocean.d.da.getRanges()
            self.ocean.d.vecArray[i0:i1,j0:j1] = 4000
            self.ocean.d.local_to_local()
            self.ocean.add_raised_cosine_xy(variable=self.ocean.d, 
                lon=0, lat=+6, amplitude=-3990, radius=200e3)
            self.ocean.add_raised_cosine_xy(variable=self.ocean.d, 
                lon=0, lat=-6, amplitude=-3990, radius=200e3)
        if False: # equally spaced atolls
            (i0, i1), (j0, j1) = self.ocean.d.da.getRanges()
            self.ocean.d.vecArray[i0:i1,j0:j1] = 4000
            self.ocean.d.local_to_local()
            for lat in range(-15,16,5):
                for lon in range(-15,16,5):
                    self.ocean.add_raised_cosine_xy(variable=self.ocean.d, 
                        #lon=lon, lat=lat, amplitude=-3990, radius=200e3)
                        lon=lon, lat=lat, amplitude=-8000, radius=200e3)
        if False: # equally spaced atolls
            (i0, i1), (j0, j1) = self.ocean.d.da.getRanges()
            self.ocean.d.vecArray[i0:i1,j0:j1] = 4000
            self.ocean.d.local_to_local()
            for lat in range(-30,31,4):
                for lon in range(-14,15,4):
                    self.ocean.add_raised_cosine_xy(variable=self.ocean.d, 
                        #lon=lon, lat=lat, amplitude=-3990, radius=200e3)
                        lon=lon, lat=lat, amplitude=-8000, radius=200e3)
        if False: # equally spaced atolls
            (i0, i1), (j0, j1) = self.ocean.d.da.getRanges()
            self.ocean.d.vecArray[i0:i1,j0:j1] = 4000
            self.ocean.d.local_to_local()
            for lat in range(-17,18,17):
                for lon in range(-17,18,17):
                    self.ocean.add_raised_cosine_xy(variable=self.ocean.d, 
                        #lon=lon, lat=lat, amplitude=-3990, radius=200e3)
                        lon=lon, lat=lat, amplitude=-8000, radius=100e3)

        if self.remove_land:
            (i0, i1), (j0, j1) = self.ocean.d.da.getRanges()
            self.ocean.d.vecArray[i0:i1,j0:j1] = np.maximum(self.remove_land, self.ocean.d.vecArray[i0:i1,j0:j1])
            self.ocean.d.local_to_local()

        taper_width = 40 #// self.RSL * 4
        self.ocean.setup(taper_width=taper_width, 
                                    station_csv="station_HaidaGwaii.csv",
                                    compressible=self.CMP)
        if False: # final model (C1S1B1, 30sec delay, obs_length 780 min - InversionOkada75.ipynb)
            self.ocean.setup_okada(lonR=360-132.247, 
                                latR=52.426, 
                                strike=314.2, 
                                depth=20.6e3, 
                                AL=[-94.2e3/2,94.2e3/2], 
                                AW=[-18.4e3,0], 
                                dip=33.4, 
                                rake=101.4)
        elif source_option=="Sheehan": # Sheehan et al
            print("Sheehan")
            self.ocean.setup_okada(lonR=360-131.7, 
                                latR=52.1, 
                                strike=314.0, 
                                depth=1.0e3, 
                                AL=[0,110.e3], 
                                AW=[-50.0e3,0], 
                                dip=25.0, 
                                rake=100.0,
                                displacement=2.5)
        elif source_option=="Gusman": # Gusman et al
            print("Gusman")
            import pandas
            df = pandas.read_csv("Gussman_param.csv", header=1)
            params={}
            params["lonR"]=list(360 + df["Lon."])
            params["latR"]=list(df["#Lat."])
            params["strike"]=list(df["strike"])
            params["depth"]=list(df["depth"] * 1e3)
            params["AL"]=[]
            params["AW"]=[]
            params["dip"]=list(df["dip"])
            params["rake"]=list(df["rake"])
            params["displacement"]=list(df["slip"])
            for i in range(len(params["lonR"])):
                params["AL"].append([0,15e3])
                params["AW"].append([-15e3,0])
            self.ocean.setup_okada(**params)
        elif source_option=="Takagawa": # Takagawa 8x4
            print("Takagawa 8x4")
            import pandas
            df = pandas.read_csv("Takagawa_param.csv", header=1)
            params={}
            params["lonR"]=list(360 + df["Lon."])
            params["latR"]=list(df["#Lat."])
            params["strike"]=list(df["strike"])
            params["depth"]=list(df["depth"] * 1e3)
            params["AL"]=[]
            params["AW"]=[]
            params["dip"]=list(df["dip"])
            params["rake"]=list(df["rake"])
            params["displacement"]=list(df["slip"])
            for i in range(len(params["lonR"])):
                params["AL"].append([0,15.984e3])
                params["AW"].append([-9.678e3,0])
            self.ocean.setup_okada(**params)
        elif source_option=="TakagawaZSlip3": # Takagawa 8x4
            print("Takagawa 8x4")
            import pandas
            df = pandas.read_csv("TakagawaZSlip3_param.csv", header=1)
            params={}
            params["lonR"]=list(360 + df["Lon."])
            params["latR"]=list(df["#Lat."])
            params["strike"]=list(df["strike"])
            params["depth"]=list(df["depth"] * 1e3)
            params["AL"]=[]
            params["AW"]=[]
            params["dip"]=list(df["dip"])
            params["rake"]=list(df["rake"])
            params["displacement"]=list(df["slip"])
            for i in range(len(params["lonR"])):
                #params["AL"].append([0,15.984e3])
                #params["AW"].append([-9.678e3,0])
                params["AL"].append([-15.984e3/2,15.984e3/2])
                params["AW"].append([ -9.678e3/2, 9.678e3/2])
            self.ocean.setup_okada(**params)
        elif source_option=="TakagawaS": # Takagawa 8x4 shallow part only
            print("Takagawa 8x4")
            import pandas
            df = pandas.read_csv("TakagawaS_param.csv", header=1)
            params={}
            params["lonR"]=list(360 + df["Lon."])
            params["latR"]=list(df["#Lat."])
            params["strike"]=list(df["strike"])
            params["depth"]=list(df["depth"] * 1e3)
            params["AL"]=[]
            params["AW"]=[]
            params["dip"]=list(df["dip"])
            params["rake"]=list(df["rake"])
            params["displacement"]=list(df["slip"])
            for i in range(len(params["lonR"])):
                params["AL"].append([0,15.984e3])
                params["AW"].append([-9.678e3,0])
            self.ocean.setup_okada(**params)
        elif source_option=="TakagawaD": # Takagawa 8x4 deep part only
            print("Takagawa 8x4")
            import pandas
            df = pandas.read_csv("TakagawaD_param.csv", header=1)
            params={}
            params["lonR"]=list(360 + df["Lon."])
            params["latR"]=list(df["#Lat."])
            params["strike"]=list(df["strike"])
            params["depth"]=list(df["depth"] * 1e3)
            params["AL"]=[]
            params["AW"]=[]
            params["dip"]=list(df["dip"])
            params["rake"]=list(df["rake"])
            params["displacement"]=list(df["slip"])
            for i in range(len(params["lonR"])):
                params["AL"].append([0,15.984e3])
                params["AW"].append([-9.678e3,0])
            self.ocean.setup_okada(**params)
        elif source_option=="TakagawaHalfS": # Takagawa 8x4 deep part only
            print("Takagawa 8x4")
            import pandas
            df = pandas.read_csv("TakagawaHalfS_param.csv", header=1)
            params={}
            params["lonR"]=list(360 + df["Lon."])
            params["latR"]=list(df["#Lat."])
            params["strike"]=list(df["strike"])
            params["depth"]=list(df["depth"] * 1e3)
            params["AL"]=[]
            params["AW"]=[]
            params["dip"]=list(df["dip"])
            params["rake"]=list(df["rake"])
            params["displacement"]=list(df["slip"])
            for i in range(len(params["lonR"])):
                params["AL"].append([0,15.984e3])
                params["AW"].append([-9.678e3,0])
            self.ocean.setup_okada(**params)
        elif source_option=="TakagawaHalfD": # Takagawa 8x4 deep part only
            print("Takagawa 8x4")
            import pandas
            df = pandas.read_csv("TakagawaHalfD_param.csv", header=1)
            params={}
            params["lonR"]=list(360 + df["Lon."])
            params["latR"]=list(df["#Lat."])
            params["strike"]=list(df["strike"])
            params["depth"]=list(df["depth"] * 1e3)
            params["AL"]=[]
            params["AW"]=[]
            params["dip"]=list(df["dip"])
            params["rake"]=list(df["rake"])
            params["displacement"]=list(df["slip"])
            for i in range(len(params["lonR"])):
                params["AL"].append([0,15.984e3])
                params["AW"].append([-9.678e3,0])
            self.ocean.setup_okada(**params)
        else: # simplified Gusman model
            print("simplified Gusman")
            self.ocean.setup_okada(lonR=360-131.5, 
                                latR=52, 
                                strike=317, 
                                depth=1e3, 
                                AL=[0,165e3], 
                                AW=[-60e3,0], 
                                dip=18.5, 
                                rake=103.3)
        # print("===== station.dataframe rank=", self.rank, "=====")
        # print(self.ocean.station.dataframe)

        dmax = self.ocean.get_max(self.ocean.d)
        lx = self.ocean.dx * self.ocean.R * np.min(np.cos(self.ocean.yN))
        ly = self.ocean.dy * self.ocean.R
        #lmin = np.minimum(lx, ly)#np.sqrt(lx**2 + ly**2)
        #gmax = np.max(self.ocean.gN)
        #self.dt = 0.99 * lmin / np.sqrt(2 * gmax * dmax)
        lmin = lx * ly / (lx + ly)#np.sqrt(lx**2 + ly**2)
        gmax = np.max(self.ocean.gN)
        self.dt = 0.99 * lmin / np.sqrt(gmax * dmax)
        if self.resolution==120:
            self.dt = np.minimum(self.dt, 3.0)
        elif self.resolution==60:
            self.dt = np.minimum(self.dt, 1.5)
        elif self.resolution==30:
            self.dt = np.minimum(self.dt, 0.75)
        #"""
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("dx   ", self.ocean.dx)
        print("dy   ", self.ocean.dy)
        print("R    ", self.ocean.R)
        print("cos  ", np.min(np.cos(self.ocean.yN)))
        print("lmin ", lmin)
        print("dt   ", self.dt, "dtmax", lmin / np.sqrt(gmax * dmax))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", flush=True)
        #"""

        if self.CLV > 0:
            sh_n = 675 # int(np.round(675 * 2**(5 - self.CLV)))
            earth = SolidEarth(self.ocean, GID=True, sh_n=sh_n, IGF=False)
            self.ocean.set_solid_earth(earth)
            print("CLV=", self.CLV, earth)

    @decompi.finalize
    def depth_random(self, x, y):
        return np.random.rand(x.shape)

    @decompi.finalize
    def main(self):
        #print(self.ocean.station.get_list_ij())
        #print(self.ocean.station.get_list_stats())
        #i_src_g = np.zeros([self.comm.Get_size(),1], dtype=np.int8)
        #j_src_g = np.zeros([self.comm.Get_size(),1], dtype=np.int8)
        
        i_src = j_src = -1
        if self.job_name == "forward":
            index_src = 0
        else:
            #index_src = 1
            index_src = int(self.job_name[7:])
        for _, row in self.ocean.station.dataframe.iterrows():
            if row.station==index_src:
                i_src = row.i
                j_src = row.j

        #print("%%%%%%%%%%%",self.comm.allgather(i_src))
        #self.comm.Allgather([i_src,  MPI.INT], [i_src_g, MPI.INT])
        #self.comm.Allgather([j_src,  MPI.INT], [j_src_g, MPI.INT])
        i_src = np.max(self.comm.allgather(i_src))
        j_src = np.max(self.comm.allgather(j_src))
        #j_src = np.max(j_src_g)
        print(index_src, i_src, j_src)
        """
        if self.job_name == "forward":
            i_src = 240#360
        else:
            i_src = 720#600
        j_src = self.NY // 2
        """
        #self.ocean.add(variable=self.ocean.h, i=i_src, j=j_src, amount=1e-6)
        amplitude = 1.0 #1e-6
        if self.job_name[:7] == "forward":
            if False:
                self.ocean.add_raised_cosine(variable=self.ocean.h, 
                                            i=i_src, j=j_src, amplitude=amplitude, radius=200e3)
            if False:
                self.ocean.add_raised_cosine_xy(variable=self.ocean.h, 
                                            lon=-24, lat=0, amplitude=amplitude, radius=100e3)
            if self.job_name == "forward":
                self.ocean.add_okada(displacement=1.0*amplitude) # changed to get waveforms of model S & G with resolution of 60 & 30 sec
            else:
                self.ocean.add_numerical_delta(variable=self.ocean.h, i=i_src, j=j_src, amplitude=amplitude)
                
        else:
            self.ocean.add_numerical_delta(variable=self.ocean.h, i=i_src, j=j_src, amplitude=amplitude)
            #self.ocean.add_raised_cosine(variable=self.ocean.h, 
            #                            i=i_src, j=j_src, amplitude=amplitude, radius=200e3)
        #self.ocean.save_xr_dataset_in_parallel("tandem_stag_0")
        time_integration_type = "long"
        if time_integration_type == "long":
            step_max = int((3600 * 9) / self.dt)  + 1
            step_interval = int(np.round(30 / self.dt))
            save_interval = int(np.round(3600 / self.dt)) 
        elif time_integration_type == "mid":
            step_max = int((3600 * 3) / self.dt)  + 1
            step_interval = int(np.round(30 / self.dt))
            save_interval = int(np.round(3600 / self.dt)) 
        else:
            step_max = 5000 #1200
            step_interval = 1
            save_interval = 50 
            
        #self.dt = 600. / 8 #/ np.sqrt(7000)
        is_adjoint = self.job_name[:7] == "adjoint"
        if self.ocean.has_Boussinesq:
            self.ocean.setup_ksp(is_adjoint=is_adjoint)
        self.ocean.station.setup_logger(step_max, self.dt, starttime="2000-01-01T00:00:00.000000Z")
        ij_list = self.ocean.station.get_list_ij()
        # set up recorder
        attrs_d = {"units": "m", "standard_name":"depth", "long_name":"water depth"}
        attrs_h = {"units": "m", "standard_name":"elevation", "long_name":"water surface elevation"}
        attrs_M = {"units": "$m^3/s$", "standard_name":"flux", "long_name":"eastward flux"}
        attrs_N = {"units": "$m^3/s$", "standard_name":"flux", "long_name":"southward flux"}

        time_h = np.arange(0, save_interval, step_interval) * self.dt
        time_MN = time_h + 0.5 * self.dt 
        xds_record_d = xr.Dataset({"d":self.ocean.get_xr_data_array_recorder(self.ocean.d, [0], attrs=attrs_d)})
        xds_record_d.d[0] = self.ocean.get_local_array(self.ocean.d)
        xds_record_d.to_netcdf(self.outpath + f"/record_d_{self.rank:04}.nc")         
        xds_record_h = xr.Dataset({"h":self.ocean.get_xr_data_array_recorder(self.ocean.h, time_h, attrs=attrs_h)})
        #xds_record_M = xr.Dataset({"M":self.ocean.get_xr_data_array_recorder(self.ocean.M, time_MN, attrs=attrs_M)})
        #xds_record_N = xr.Dataset({"N":self.ocean.get_xr_data_array_recorder(self.ocean.N, time_MN, attrs=attrs_N)})
        for step in range(step_max):
            #if self.rank==0:
            #    print(step, end=" ", flush=True)
            #hmax = self.ocean.get_max(self.ocean.h)
            #hmin = self.ocean.get_min(self.ocean.h)
            #if self.rank==0:
            #    print(hmax)
            if step % save_interval==0:
                save_steps = step + np.arange(0, save_interval, step_interval)
                time_h = save_steps * self.dt
                time_MN = time_h + 0.5 * self.dt 
                xds_record_h = xr.Dataset({"h":self.ocean.get_xr_data_array_recorder(self.ocean.h, time_h, attrs=attrs_h)})
                #xds_record_M = xr.Dataset({"M":self.ocean.get_xr_data_array_recorder(self.ocean.M, time_MN, attrs=attrs_M)})
                #xds_record_N = xr.Dataset({"N":self.ocean.get_xr_data_array_recorder(self.ocean.N, time_MN, attrs=attrs_N)})
            #values = self.ocean.h.get_values(ij_list)
            #print(f"ij_list:{ij_list}")
            values = self.ocean.get_filtered_h(ij_list)
            self.ocean.station.record(step, values)
            if step % step_interval==0:
                idx_chunk = (step % save_interval) // step_interval
                xds_record_h.h[idx_chunk] = self.ocean.get_local_array(self.ocean.h) # record the value
                #xds_record_M.M[idx_chunk] = self.ocean.get_local_array(self.ocean.M) # record the value
                #xds_record_N.N[idx_chunk] = self.ocean.get_local_array(self.ocean.N) # record the value
            if (step + 1) % save_interval==0:
                idx_save = step // save_interval
                output_small_area = True
                if output_small_area: # output small area
                    lon0 = 360 - 132.25
                    lat0 = 52.5
                    xmin = float(xds_record_h.longitude.min()) -4
                    xmax = float(xds_record_h.longitude.max()) +4
                    ymin = float(xds_record_h.latitude.min()) -4
                    ymax = float(xds_record_h.latitude.max()) +4
                    if xmin<lon0<xmax and ymin<lat0<ymax:
                        xds_record_h.to_netcdf(self.outpath + f"/record_h_{self.rank:04}_{idx_save:03}.nc") # save the record
                else:
                    xds_record_h.to_netcdf(self.outpath + f"/record_h_{self.rank:04}_{idx_save:03}.nc") # save the record
                    #xds_record_M.to_netcdf(self.outpath + f"/record_M_{self.rank:04}_{idx_save:03}.nc") # save the record
                    #xds_record_N.to_netcdf(self.outpath + f"/record_N_{self.rank:04}_{idx_save:03}.nc") # save the record
        
            #print(xda_record_h[step])
            COR=True
            SPG=True
            SMF=True
            if self.job_name[:7] == "forward":
                self.ocean.forward(+self.dt, with_Coriolis=COR, is_reversal=False, with_sponge=SPG, with_Sommerfeld=SMF)
            elif self.job_name[:7] == "reverse":
                self.ocean.forward(+self.dt, with_Coriolis=COR, is_reversal=True, with_sponge=SPG, with_Sommerfeld=SMF)
                # self.ocean.forward(-self.dt, with_Coriolis=COR, is_reversal=False, with_sponge=SPG, with_Sommerfeld=SMF)
            elif self.job_name[:7] == "adjoint":
                self.ocean.adjoint(+self.dt, with_Coriolis=COR, with_sponge=SPG, with_Sommerfeld=SMF)
            hmax = self.ocean.get_max(self.ocean.h)
            hmin = self.ocean.get_min(self.ocean.h)
            if self.rank==0:
                print(f"{step} {hmax:10.2f} {hmin:10.2f}")
            i,j=self.ocean.h.get_slice()
            #if hmax>20 and np.max(self.ocean.h.vecArray[i[0],j[0]])==hmax:
            #    print("++++++++++++++ hmax > 20 ++++++++++++++++++++++")
            #    print(self.rank, hmax, np.argmax(self.ocean.h.vecArray[i[0],j[0]]) , i[0].start, i[0].stop, j[0].start, j[0].stop)
            #    break
            #if self.ocean.get_max(self.ocean.h)>1:
            #    self.ocean.save_xr_dataset_in_parallel("overflow")
            #    break
        #self.ocean.save_xr_dataset_in_parallel("tandem_stag_1")
        xds_record_hmax = xr.Dataset({"hmax":self.ocean.get_xr_data_array_recorder(self.ocean.hmax, [0], attrs=attrs_d)})
        xds_record_hmax.hmax[0] = self.ocean.get_local_array(self.ocean.hmax)
        xds_record_hmax.to_netcdf(self.outpath + f"/record_hmax_{self.rank:04}.nc")         

        self.ocean.station.save_timeseries(self.outpath + f"/timeseries{self.rank:04}.mseed")
        #print(index_src, i_src, j_src)
        """
        print("\n===== mid =====", flush=True)
        step_max = 0  # do not run the adjoint process
        if False:
            self.ocean.setup_ksp(is_adjoint=True, with_Coriolis=True, with_sponge=True, with_Sommerfeld=True)
            for step in range(step_max):
                print(step, end=" ")
                self.ocean.adjoint(+self.dt)
        else:
            for step in range(step_max):
                print(step, end=" ")
                self.ocean.forward(-self.dt)
        if self.rank==0:
            print("time for fwd & adj:", time.time() - time_start)
        self.ocean.save_xr_dataset_in_parallel("tandem_stag_2")
        """

def get_job_info():
    print("args:", sys.argv)
    args = sys.argv
    outpath = ""
    job_id = ""
    job_name = ""
    if len(args) > 1:
        if os.path.exists(args[1]):
            outpath = args[1]
            job_id = args[1].split("-")[1]
            job_name = args[1].split("-")[2]
    return (outpath, job_id, job_name)

if __name__=="__main__":
    outpath, job_id, job_name = get_job_info()
    if outpath == "":
        tandem = Tandem()
    else:
        print("outpath:", outpath)
        print("job_id:", job_id)
        print("job_name:", job_name, flush=True)
        tandem = Tandem(outpath, job_id, job_name)
    rank = MPI.COMM_WORLD.rank
    if rank==0:
        os.mkdir(f"{outpath}/profile")
    filename = f'{outpath}/profile/{job_id}-{job_name}-{rank:04}.prof'
    print(cProfile)
    cProfile.run('tandem.main()', filename=filename)
