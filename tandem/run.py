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

import sys
import os
import numpy as np
from mpi4py import MPI
from tandem.ocean import Ocean
from tandem.solidearth import SolidEarth
import xarray as xr
from tandem.decompi import DecoratorMPI
import cProfile
import pandas
import shutil
import json

decompi = DecoratorMPI()


settings = dict(
    # output
    output_directory="out",
    save_params="hMN", # ["", "h", "hMN"]

    #calc condition
    integration="forward", # ["forward", "backward", "adjoint"]
    compressibility=True,
    SAL=True, # Self-Attraction and Loading effect
    Boussinesq=False,
    Coriolis=True,
    advection=False,
    sht_reduction= "auto", # ["auto", 1-5]
    resolution= 480, # [15, 30, 60, 120, 240, 480 sec]
    Manning=0.0, # 0.025
    sponge=dict(
        width=40,
        damping_factor=0.01
        ),
    open_boundary=True,
    grid=dict(
        west=164, # [deg]
        east=240, # [deg]
        north=64, # [deg]
        south=-4, # [deg]
        resolution=480 # [sec]
        ),
    
    # input files
    topo=dict(
        file="data/gebco2023_8min_median/GEBCO*.nc",
        depth = False,  # elevation --> False / depth --> True
        lon = "lon",    # variable name in the file 
        lat = "lat",    # variable name in the file
        z = "elevation" # variable name in the file
        ),
    station=dict(
        file="data/station.csv", 
        header=1
        ),
    point=dict(
        station_id=11,
        source=False
        ),
    fault=dict(
        file="data/fault.csv",
        header=1,
        source=True,
        ),
    
    # time
    t_max = 3600 * 7, # [sec]
    rec_interval_time = 60 * 10, # [sec]
    dt = "auto",
    chunk_size = 6, # number of data in a chunk
    
    )



        


class Tandem(object):
    @decompi.finalize
    def __init__(self,):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.outpath = settings["output_directory"]
        self.save_values = settings["save_params"]
        if self.rank==0:
            if os.path.exists(self.outpath):
                shutil.rmtree(self.outpath)
            os.makedirs(self.outpath, exist_ok=True)
            os.makedirs(self.outpath + "/d", exist_ok=True)
            if "h" in self.save_values:
                os.makedirs(self.outpath + "/h", exist_ok=True)
            if "M" in self.save_values:
                os.makedirs(self.outpath + "/M", exist_ok=True)
            if "N" in self.save_values:
                os.makedirs(self.outpath + "/N", exist_ok=True)
            os.makedirs(self.outpath + "/hmax", exist_ok=True)
        _dsh = 180 * 60 // 675 # [arcmin] minimum mesh size to apply shtns
        _wr = settings["sponge"]["width"] * settings["grid"]["resolution"] / 3600
        nW = int(np.floor((settings["grid"]["west" ] - _wr) / (_dsh / 60)))       #############南北方向のメッシュ数がアジョイントモデルの時に２つ食い違う。赤道を原点として割り切れる位置を端に設定するのではなく、北極を原点として割り切れる位置にする。180度を675分割しているが、これは90度の赤道を通らない？
        nE = int(np.ceil ((settings["grid"]["east" ] + _wr) / (_dsh / 60)))
        nN = int(np.ceil ((settings["grid"]["north"] + _wr) / (_dsh / 60)))
        nS = int(np.floor((settings["grid"]["south"] - _wr) / (_dsh / 60)))
        settings["grid"]["west" ] = nW * (_dsh / 60)
        settings["grid"]["east" ] = nE * (_dsh / 60)
        settings["grid"]["north"] = nN * (_dsh / 60)
        settings["grid"]["south"] = nS * (_dsh / 60)
        self.extent = np.deg2rad([settings["grid"]["west" ], 
                                  settings["grid"]["east" ], 
                                  settings["grid"]["north"], 
                                  settings["grid"]["south"]]) 
        self.NX = (nE - nW) * _dsh * 60 // settings["grid"]["resolution"]
        self.NY = (nN - nS) * _dsh * 60 // settings["grid"]["resolution"]

        self.shape = (self.NX, self.NY)
        if settings["SAL"]:
            if settings["grid"]["resolution"] in [15,30,60,120,240,480] and settings["sht_reduction"]=="auto":
                settings["sht_reduction"] = int(np.log2(480 // settings["grid"]["resolution"])) + 1
            else:
                raise ValueError("resolution must be one of 15, 30, 60, 120, 240 and 480 arc seconds.")
        else:
            settings["sht_reduction"] = 0
        
        self.ocean = Ocean(self.shape, extent=self.extent, damping_factor=settings["sponge"]["damping_factor"],
                            CLV=settings["sht_reduction"], 
                            has_Boussinesq=settings["Boussinesq"],
                            outpath=settings["output_directory"],  
                            Manning=settings["Manning"], 
                            Nonlinear=["advection"])
        self.ocean.load_bathymetry(settings["topo"]["file"], 
                                   depth=settings["topo"]["depth"], 
                                   lon=settings["topo"]["lon"], 
                                   lat=settings["topo"]["lat"], 
                                   z=settings["topo"]["z"])
        
        # record d
        attrs_d = {"units": "m", "standard_name":"depth", "long_name":"water depth"}
        xds_record_d = xr.Dataset({"d":self.ocean.get_xr_data_array_recorder(self.ocean.d, [0], attrs=attrs_d)})
        xds_record_d.d[0] = self.ocean.get_local_array(self.ocean.d)
        xds_record_d.isel(time=0).transpose().to_netcdf(self.outpath + f"/d/d_{self.rank:04}.nc")         

        self.ocean.remove_land_elevation()

        self.ocean.setup(taper_width=settings["sponge"]["width"], 
                                    station_csv=settings["station"]["file"],
                                    compressible=settings["compressibility"])

        df = pandas.read_csv(settings["fault"]["file"], 
                             header=settings["fault"]["header"])
        params={}
        params["lonR"]=list(360 + df["Lon."])
        params["latR"]=list(df["Lat."])
        params["strike"]=list(df["strike"])
        params["depth"]=list(df["depth"] * 1e3)
        params["AL"]=[[ 0    , L*1e3] for L in df["L"]]
        params["AW"]=[[-W*1e3, 0    ] for W in df["W"]]
        params["dip"]=list(df["dip"])
        params["rake"]=list(df["rake"])
        params["displacement"]=list(df["slip"])
        self.ocean.setup_okada(**params)

        dmax = self.ocean.get_max(self.ocean.d)
        print("dmax", dmax)
        lx = self.ocean.dx * self.ocean.R * np.min(np.cos(self.ocean.yN))
        ly = self.ocean.dy * self.ocean.R
        lmin = lx * ly / (lx + ly) #np.sqrt(lx**2 + ly**2)
        #lmin = np.sqrt(lx**2 + ly**2)
        gmax = np.max(self.ocean.gN)
        self.dt = 0.99 * lmin / np.sqrt(gmax * dmax)
        
        t_max = settings["t_max"] # [sec]
        self.rec_interval_time = settings["rec_interval_time"] # [sec]
        self.rec_interval_step = int(np.ceil(self.rec_interval_time / self.dt))
        self.dt = self.rec_interval_time / self.rec_interval_step
        settings["dt"] = self.dt
        self.step_max = int(np.ceil(t_max / self.dt))
        self.chunk_size = settings["chunk_size"] # number of data in a chunk
        self.chunk_step = self.chunk_size * self.rec_interval_step

        if self.rank==0:
            print("dt   ", self.dt, "dtmax", lmin / np.sqrt(gmax * dmax))
            
        if settings["sht_reduction"] > 0:
            sh_n = 675 # int(np.round(675 * 2**(5 - self.CLV)))
            earth = SolidEarth(self.ocean, GID=True, sh_n=sh_n, IGF=False)
            self.ocean.set_solid_earth(earth)

    @decompi.finalize
    def main(self):
        
        i_src = j_src = -1
        if settings["integration"] == "forward":
            index_src = 0
        else:
            settings["fault"]["source"] = False
            settings["point"]["source"] = True
            index_src = int(settings["point"]["station_id"])
        for _, row in self.ocean.station.dataframe.iterrows():
            if row.station==index_src:
                i_src = row.i
                j_src = row.j

        i_src = np.max(self.comm.allgather(i_src))
        j_src = np.max(self.comm.allgather(j_src))
        amplitude = 1#1e-6
        if settings["integration"] == "forward":
            if settings["fault"]["source"]:
                self.ocean.add_okada(displacement=1.0*amplitude)
            if settings["point"]["source"]:
                self.ocean.add_numerical_delta(variable=self.ocean.h, i=i_src, j=j_src, amplitude=amplitude)
        else:
            self.ocean.add_numerical_delta(variable=self.ocean.h, i=i_src, j=j_src, amplitude=amplitude)
            
        if self.ocean.has_Boussinesq:
            self.ocean.setup_ksp(is_adjoint=settings["integration"]=="adjoint")
        self.ocean.station.setup_logger(self.step_max, self.dt, starttime="2000-01-01T00:00:00.000000Z")
        ij_list = self.ocean.station.get_list_ij()
        
        # set up attrs
        attrs_h = {"units": "m", "standard_name":"elevation", "long_name":"water surface elevation"}
        attrs_M = {"units": "$m^3/s$", "standard_name":"flux", "long_name":"eastward flux"}
        attrs_N = {"units": "$m^3/s$", "standard_name":"flux", "long_name":"southward flux"}
        attrs_hmax = {"units": "m", "standard_name":"elevation", "long_name":"maximum water surface height"}
        
        # record settings
        if self.rank==0:
            with open(f'{settings["output_directory"]}/settings.json', "w") as f:
                json.dump(settings, f, indent=2)

        # main loop
        for step in range(self.step_max):

            # record time series data at stations
            values = self.ocean.get_hMNUV(ij_list)
            self.ocean.station.record(step, step*self.dt, values)

            # initialize chunk xds_racords
            if step % self.chunk_step==0:
                save_steps = step + np.arange(self.chunk_size) * self.rec_interval_step
                time_h = save_steps * self.dt
                time_MN = time_h + 0.5 * self.dt 
                if "h" in self.save_values:
                    xds_record_h = xr.Dataset({"h":self.ocean.get_xr_data_array_recorder(self.ocean.h, time_h, attrs=attrs_h)})
                if "M" in self.save_values:
                    xds_record_M = xr.Dataset({"M":self.ocean.get_xr_data_array_recorder(self.ocean.M, time_MN, attrs=attrs_M)})
                if "N" in self.save_values:
                    xds_record_N = xr.Dataset({"N":self.ocean.get_xr_data_array_recorder(self.ocean.N, time_MN, attrs=attrs_N)})
            # record values
            if step % self.rec_interval_step==0:
                idx_chunk = (step % self.chunk_step) // self.rec_interval_step
                if "h" in self.save_values:
                    xds_record_h.h[idx_chunk] = self.ocean.get_local_array(self.ocean.h)
                if "M" in self.save_values:
                    xds_record_M.M[idx_chunk] = self.ocean.get_local_array(self.ocean.M)
                if "N" in self.save_values:
                    xds_record_N.N[idx_chunk] = self.ocean.get_local_array(self.ocean.N)
            # save chunks
            if (step + 1) % self.chunk_step==0:
                idx_save = step // self.chunk_step
                if "h" in self.save_values:
                    xds_record_h.transpose("time", "latitude", "longitude")\
                        .to_netcdf(self.outpath + f"/h/h_{self.rank:04}_{idx_save:03}.nc")
                if "M" in self.save_values:
                    xds_record_M.transpose("time", "latitude", "longitude")\
                        .to_netcdf(self.outpath + f"/M/M_{self.rank:04}_{idx_save:03}.nc")
                if "N" in self.save_values:
                    xds_record_N.transpose("time", "latitude", "longitude")\
                        .to_netcdf(self.outpath + f"/N/N_{self.rank:04}_{idx_save:03}.nc")
            # time integration
            if settings["integration"] == "forward":
                self.ocean.forward(+self.dt, 
                                   with_Coriolis=settings["Coriolis"], 
                                   is_reversal=False, 
                                   with_sponge=settings["sponge"]["width"]>0, 
                                   with_Sommerfeld=settings["open_boundary"])
            elif settings["integration"] == "backward":
                self.ocean.forward(+self.dt, 
                                   with_Coriolis=settings["Coriolis"], 
                                   is_reversal=True , 
                                   with_sponge=settings["sponge"]["width"]>0, 
                                   with_Sommerfeld=settings["open_boundary"])
            elif settings["integration"] == "adjoint":
                self.ocean.adjoint(+self.dt, 
                                   with_Coriolis=settings["Coriolis"], 
                                   with_sponge=settings["sponge"]["width"]>0, 
                                   with_Sommerfeld=settings["open_boundary"])
            if step % 10 ==0:
                hmax = self.ocean.get_max(self.ocean.h)
                hmin = self.ocean.get_min(self.ocean.h)
                if self.rank==0:
                    print(f"{step}/{self.step_max} {hmax:10.2f} {hmin:10.2f}", end="\t")
            else:
                if self.rank==0:
                    print(".", end="")
            if (step + 1) % 10 ==0:
                if self.rank==0:
                    print()
        # save hmax
        xds_record_hmax = xr.Dataset({"hmax":self.ocean.get_xr_data_array_recorder(self.ocean.hmax, [0], attrs=attrs_hmax)})
        xds_record_hmax.hmax[0] = self.ocean.get_local_array(self.ocean.hmax)
        xds_record_hmax.isel(time=0).transpose()\
            .to_netcdf(self.outpath + f"/hmax/hmax_{self.rank:04}.nc")
        
        self.ocean.station.save_timeseries(self.outpath)
        if self.rank==0:
            print("\n===== finish! =====")

if __name__=="__main__":
    outpath="out"
    tandem = Tandem()
    rank = MPI.COMM_WORLD.rank
    if rank==0:
        os.makedirs(f"{outpath}/profile", exist_ok=True)
    filename = f'{outpath}/profile/{rank:04}.prof'
    cProfile.run('tandem.main()', filename=filename)
