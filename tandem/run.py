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
from mpi4py import MPI
import xarray as xr
from tandem.decompi import DecoratorMPI
import cProfile
import pandas
import shutil

decompi = DecoratorMPI()

class Tandem(object):
    @decompi.finalize
    def __init__(self, settings="settings.json"):
        outpath="outC0S0B0"
        job_id="0000000"
        job_name="forward_C0S0B0GR480_TakagawaZSlip3_Manning0.000"
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.outpath = outpath
        self.save_values = "hMN"

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
        self.job_id = job_id
        self.Nonlinear = "_NL" in job_name
        #print("Nonlinear:", self.Nonlinear, flush=True)
        self.job_name, job_option, source_option, manning_option = job_name.split("_")[:4]
        #print("manning_option:", manning_option, flush=True)
        self.CMP = "C1" in job_option
        self.SAL = "S1" in job_option
        self.BSQ = "B1" in job_option
        self.resolution = 8 * 60 # arcsec
        Manning = float(manning_option[7:]) # Manning 0.025

        taper_width = 40
        westend, eastend, northend, southend = [164, 240, 64, -4] # [degree]
        _dsh = 180 * 60 // 675 # [arcmin] minimum mesh size to apply shtns
        nW = int(np.floor((westend  - taper_width * self.resolution / 3600) / (_dsh / 60)))
        nE = int(np.ceil ((eastend  + taper_width * self.resolution / 3600) / (_dsh / 60)))
        nN = int(np.ceil ((northend + taper_width * self.resolution / 3600) / (_dsh / 60)))
        nS = int(np.floor((southend - taper_width * self.resolution / 3600) / (_dsh / 60)))
        westend  = nW * (_dsh / 60)
        eastend  = nE * (_dsh / 60)
        northend = nN * (_dsh / 60)
        southend = nS * (_dsh / 60)
        self.extent = np.deg2rad([westend, eastend, northend, southend]) 
        self.NX = (nE - nW) * _dsh * 60 // self.resolution
        self.NY = (nN - nS) * _dsh * 60 // self.resolution

        self.shape = (self.NX, self.NY)
        if self.rank==0:
            print(westend, eastend, northend, southend, self.NX, self.NY)
        if self.SAL:
            if self.resolution in [15,30,60,120,240,480]:
                self.CLV = int(np.log2(480 // self.resolution)) + 1
            else:
                raise ValueError("resolution must be one of 15, 30, 60, 120, 240 and 480 arc seconds.")
        else:
            self.CLV = 0
        
        self.ocean = Ocean(self.shape, extent=self.extent, damping_factor=0.01,
                            CLV=self.CLV, has_Boussinesq=self.BSQ,
                            outpath=self.outpath,  
                            Manning=Manning, Nonlinear=self.Nonlinear)
        filename = "data/gebco2023_8min_median/GEBCO*.nc"
        self.ocean.load_bathymetry(filename, depth=False, lon="lon", lat="lat", z="elevation")
        
        
        self.ocean.setup(taper_width=taper_width, 
                                    station_csv="data/station.csv",
                                    compressible=self.CMP)

        df = pandas.read_csv("data/fault_param.csv", header=1)
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
            params["AL"].append([-15.984e3/2,15.984e3/2])
            params["AW"].append([ -9.678e3/2, 9.678e3/2])
        self.ocean.setup_okada(**params)

        dmax = self.ocean.get_max(self.ocean.d)
        lx = self.ocean.dx * self.ocean.R * np.min(np.cos(self.ocean.yN))
        ly = self.ocean.dy * self.ocean.R
        lmin = lx * ly / (lx + ly) #np.sqrt(lx**2 + ly**2)
        #lmin = np.sqrt(lx**2 + ly**2)
        gmax = np.max(self.ocean.gN)
        self.dt = 0.99 * lmin / np.sqrt(gmax * dmax)
        
        t_max = 3600 * 3 # [sec]
        self.rec_interval_time = 60 * 10 # [sec]
        self.rec_interval_step = int(np.ceil(self.rec_interval_time / self.dt))
        self.dt = self.rec_interval_time / self.rec_interval_step
        self.step_max = int(np.ceil(t_max / self.dt))
        self.chunk_size = 6 # number of data in a chunk
        self.chunk_step = self.chunk_size * self.rec_interval_step

        if self.rank==0:
            print("dt   ", self.dt, "dtmax", lmin / np.sqrt(gmax * dmax))
            
        if self.CLV > 0:
            sh_n = 675 # int(np.round(675 * 2**(5 - self.CLV)))
            earth = SolidEarth(self.ocean, GID=True, sh_n=sh_n, IGF=False)
            self.ocean.set_solid_earth(earth)

    @decompi.finalize
    def main(self):
        
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

        i_src = np.max(self.comm.allgather(i_src))
        j_src = np.max(self.comm.allgather(j_src))
        amplitude = 1#1e-6
        if self.job_name[:7] == "forward":
            if self.job_name == "forward":
                self.ocean.add_okada(displacement=1.0*amplitude) # changed to get waveforms of model S & G with resolution of 60 & 30 sec
            else:
                self.ocean.add_numerical_delta(variable=self.ocean.h, i=i_src, j=j_src, amplitude=amplitude)
                
        else:
            self.ocean.add_numerical_delta(variable=self.ocean.h, i=i_src, j=j_src, amplitude=amplitude)
            
        is_adjoint = self.job_name[:7] == "adjoint"
        if self.ocean.has_Boussinesq:
            self.ocean.setup_ksp(is_adjoint=is_adjoint)
        self.ocean.station.setup_logger(self.step_max, self.dt, starttime="2000-01-01T00:00:00.000000Z")
        ij_list = self.ocean.station.get_list_ij()
        # set up recorder
        attrs_d = {"units": "m", "standard_name":"depth", "long_name":"water depth"}
        attrs_h = {"units": "m", "standard_name":"elevation", "long_name":"water surface elevation"}
        attrs_M = {"units": "$m^3/s$", "standard_name":"flux", "long_name":"eastward flux"}
        attrs_N = {"units": "$m^3/s$", "standard_name":"flux", "long_name":"southward flux"}
        attrs_hmax = {"units": "m", "standard_name":"elevation", "long_name":"maximum water surface height"}
        
        
        #time_h = np.arange(self.stack_size) * self.dt
        #time_MN = time_h + 0.5 * self.dt 
        xds_record_d = xr.Dataset({"d":self.ocean.get_xr_data_array_recorder(self.ocean.d, [0], attrs=attrs_d)})
        xds_record_d.d[0] = self.ocean.get_local_array(self.ocean.d)
        xds_record_d.isel(time=0).transpose().to_netcdf(self.outpath + f"/d/d_{self.rank:04}.nc")         
        """if "h" in self.save_values:
            xds_record_h = xr.Dataset({"h":self.ocean.get_xr_data_array_recorder(self.ocean.h, time_h, attrs=attrs_h)})
        if "M" in self.save_values:
            xds_record_M = xr.Dataset({"M":self.ocean.get_xr_data_array_recorder(self.ocean.M, time_MN, attrs=attrs_M)})
        if "N" in self.save_values:
            xds_record_N = xr.Dataset({"N":self.ocean.get_xr_data_array_recorder(self.ocean.N, time_MN, attrs=attrs_N)})"""
        for step in range(self.step_max):
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
            values = self.ocean.get_hMNUV(ij_list)
            self.ocean.station.record(step, step*self.dt, values)
            if step % self.rec_interval_step==0:
                idx_chunk = (step % self.chunk_step) // self.rec_interval_step
                if "h" in self.save_values:
                    xds_record_h.h[idx_chunk] = self.ocean.get_local_array(self.ocean.h) # record the value
                if "M" in self.save_values:
                    xds_record_M.M[idx_chunk] = self.ocean.get_local_array(self.ocean.M) # record the value
                if "N" in self.save_values:
                    xds_record_N.N[idx_chunk] = self.ocean.get_local_array(self.ocean.N) # record the value
            if (step + 1) % self.chunk_step==0:
                idx_save = step // self.chunk_step
                if "h" in self.save_values:
                    xds_record_h.transpose("time", "latitude", "longitude")\
                        .to_netcdf(self.outpath + f"/h/h_{self.rank:04}_{idx_save:03}.nc") # save the record
                if "M" in self.save_values:
                    xds_record_M.transpose("time", "latitude", "longitude")\
                        .to_netcdf(self.outpath + f"/M/M_{self.rank:04}_{idx_save:03}.nc") # save the record
                if "N" in self.save_values:
                    xds_record_N.transpose("time", "latitude", "longitude")\
                        .to_netcdf(self.outpath + f"/N/N_{self.rank:04}_{idx_save:03}.nc") # save the record
        
            COR=True
            SPG=True
            SMF=True
            if self.job_name[:7] == "forward":
                self.ocean.forward(+self.dt, with_Coriolis=COR, is_reversal=False, with_sponge=SPG, with_Sommerfeld=SMF)
            elif self.job_name[:7] == "reverse":
                self.ocean.forward(+self.dt, with_Coriolis=COR, is_reversal=True, with_sponge=SPG, with_Sommerfeld=SMF)
            elif self.job_name[:7] == "adjoint":
                self.ocean.adjoint(+self.dt, with_Coriolis=COR, with_sponge=SPG, with_Sommerfeld=SMF)
            hmax = self.ocean.get_max(self.ocean.h)
            hmin = self.ocean.get_min(self.ocean.h)
            if self.rank==0:
                print(f"{step} {hmax:10.2f} {hmin:10.2f}")
            i,j=self.ocean.h.get_slice()
        xds_record_hmax = xr.Dataset({"hmax":self.ocean.get_xr_data_array_recorder(self.ocean.hmax, [0], attrs=attrs_hmax)})
        xds_record_hmax.hmax[0] = self.ocean.get_local_array(self.ocean.hmax)
        xds_record_hmax.isel(time=0).transpose()\
            .to_netcdf(self.outpath + f"/hmax/hmax_{self.rank:04}.nc")         

        self.ocean.station.save_timeseries(self.outpath + f"/timeseries{self.rank:04}.mseed")

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
    outpath="out"
    tandem = Tandem()
    rank = MPI.COMM_WORLD.rank
    if rank==0:
        os.makedirs(f"{outpath}/profile", exist_ok=True)
    filename = f'{outpath}/profile/{rank:04}.prof'
    print(cProfile)
    cProfile.run('tandem.main()', filename=filename)
