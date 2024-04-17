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
from petsc4py import PETSc
from petsc4py.PETSc import DMStag
from tandem.xslice import SliceEx

class DMDAWrapper(object):
    def __init__(self, local=False):
        self.comm = PETSc.COMM_WORLD
        self.rank = self.comm.rank
        self.local = local
        self.location = None
        self.dof_index = None
    def scatter(self):
        self.scat.scatter(self.rank0, self.naturalVec, \
            False, PETSc.Scatter.Mode.REVERSE)
        self.da.naturalToGlobal(self.naturalVec, self.globalVec)
        self.da.globalToLocal(self.globalVec,self.localVec)
    def gather(self):
        self.da.localToGlobal(self.localVec, self.globalVec)
        self.da.globalToNatural(self.globalVec, self.naturalVec)
        self.scat.scatter(self.naturalVec, self.rank0, \
            False, PETSc.Scatter.Mode.FORWARD)
    def local_to_local(self):
        self.da.localToLocal(self.localVec, self.localVec)
    def local_to_global(self):
        self.da.localToGlobal(self.localVec, self.globalVec)
    def global_to_local(self):
        self.da.globalToLocal(self.globalVec, self.localVec)
    def print_rank_zero(self, name):
        self.gather()
        if self.rank==0:
            nx, ny = self.da.getSizes()
            print(f"--- name:{name} nx:{nx} ny:{ny}----------")
            print(self.rank0[:].reshape(ny, nx), flush=True)
        self.comm.Barrier()
    def print_vec_array(self, name):
        import time
        time.sleep(0.1 * self.rank)
        print(f"--- name:{name} rank{self.rank} ----------")
        print(self.vecArray[:].T, flush=True)

    def set_boundary(self, boundary_type="copy"):
        (iMin, iMax), (jMin, jMax) = self.da.getRanges()
        nx, ny = self.da.getSizes()
        if boundary_type == "copy":
            if iMin==0:
                self.vecArray[-1,:] = self.vecArray[0,:]
            if jMin==0:
                self.vecArray[:,-1] = self.vecArray[:,0]
            if iMax==nx:
                self.vecArray[nx,:] =self.vecArray[nx-1,:]
            if jMax==ny:
                self.vecArray[:,ny] =self.vecArray[:,ny-1]
        elif boundary_type == "linear":
            if iMin==0:
                self.vecArray[-1,:] = 2 * self.vecArray[0,:] - self.vecArray[1,:]
            if jMin==0:
                self.vecArray[:,-1] = 2 * self.vecArray[:,0] - self.vecArray[:,1]
            if iMax==nx:
                self.vecArray[nx,:] = 2 * self.vecArray[nx-1,:] - self.vecArray[nx-2,:]
            if jMax==ny:
                self.vecArray[:,ny] = 2 * self.vecArray[:,ny-1] - self.vecArray[:,ny-2]
        elif boundary_type == "mirror":
            if iMin==0:
                self.vecArray[-1,:] = -self.vecArray[0,:]
            if jMin==0:
                self.vecArray[:,-1] = -self.vecArray[:,0]
            if iMax==nx:
                self.vecArray[nx,:] = -self.vecArray[nx-1,:]
            if jMax==ny:
                self.vecArray[:,ny] = -self.vecArray[:,ny-1]
        elif boundary_type == "zero":
            if iMin==0:
                self.vecArray[-1,:] = 0
            if jMin==0:
                self.vecArray[:,-1] = 0
            if iMax==nx:
                self.vecArray[nx,:] = 0
            if jMax==ny:
                self.vecArray[:,ny] = 0
        self.local_to_local()
    def set_meanx(self, src, boundary_type="none", positive_restriction=False):
        dst = self
        if boundary_type != "none":
            src.set_boundary(boundary_type=boundary_type)
        (iMin, iMax), (jMin, jMax) = dst.da.getRanges()
        nx, _ = dst.da.getSizes()
        if src.location=="element" and dst.location=="left":
            dst.vecArray[iMin:iMax,jMin:jMax] = \
                0.5 * (src.vecArray[iMin-1:iMax-1,jMin:jMax] + src.vecArray[iMin:iMax,jMin:jMax])
            if positive_restriction:
                dst.vecArray[iMin:iMax,jMin:jMax] *= (src.vecArray[iMin-1:iMax-1,jMin:jMax] > 0)
                dst.vecArray[iMin:iMax,jMin:jMax] *= (src.vecArray[iMin:iMax,jMin:jMax] > 0)
            if iMin==0:
                dst.vecArray[iMin,jMin:jMax] = dst.vecArray[iMin+1,jMin:jMax]
            if iMax==nx:
                dst.vecArray[iMax-1,jMin:jMax] = dst.vecArray[iMax-2,jMin:jMax]
            dst.local_to_local()
        elif src.location=="left" and dst.location=="element":
            dst.vecArray[iMin:iMax,jMin:jMax] = \
                0.5 * (src.vecArray[iMin+1:iMax+1,jMin:jMax] + src.vecArray[iMin:iMax,jMin:jMax])
            if positive_restriction:
                dst.vecArray[iMin:iMax,jMin:jMax] *= (src.vecArray[iMin+1:iMax+1,jMin:jMax] > 0)
                dst.vecArray[iMin:iMax,jMin:jMax] *= (src.vecArray[iMin:iMax,jMin:jMax] > 0)
            dst.local_to_local()
        else:
            message = "Error: location of dst is not suitable for src.\n"\
                      + f"src_loc={src.location}\n"\
                      + f"dst_loc={dst.location}"
            raise Exception(message)
    def set_meany(self, src, boundary_type="none", positive_restriction=False):
        dst = self
        if boundary_type != "none":
            src.set_boundary(boundary_type=boundary_type)
        (iMin, iMax), (jMin, jMax) = dst.da.getRanges()
        _, ny = dst.da.getSizes()
        if src.location=="element" and dst.location=="down":
            dst.vecArray[iMin:iMax,jMin:jMax] = \
                0.5 * (src.vecArray[iMin:iMax,jMin-1:jMax-1] + src.vecArray[iMin:iMax,jMin:jMax])
            if positive_restriction:
                dst.vecArray[iMin:iMax,jMin:jMax] *= (src.vecArray[iMin:iMax,jMin-1:jMax-1] > 0)
                dst.vecArray[iMin:iMax,jMin:jMax] *= (src.vecArray[iMin:iMax,jMin:jMax] > 0)
            if jMin==0:
                dst.vecArray[iMin:iMax,jMin] = dst.vecArray[iMin:iMax,jMin+1]
            if jMax==ny:
                dst.vecArray[iMin:iMax,jMax-1] = dst.vecArray[iMin:iMax,jMax-2]    
            dst.local_to_local()
        elif src.location=="down" and dst.location=="element":
            dst.vecArray[iMin:iMax,jMin:jMax] = \
                0.5 * (src.vecArray[iMin:iMax,jMin+1:jMax+1] + src.vecArray[iMin:iMax,jMin:jMax])
            dst.local_to_local()
        else:
            message = "Error: location of dst is not suitable for src.\n"\
                      + f"src_loc={src.location}\n"\
                      + f"dst_loc={dst.location}"
            raise Exception(message)
    def set_diffx(self, src, boundary_type="none"):
        dst = self
        if boundary_type != "none":
            src.set_boundary(boundary_type=boundary_type)
        (iMin, iMax), (jMin, jMax) = dst.da.getRanges()
        nx, ny = dst.da.getSizes()
        if src.location=="element" and dst.location=="left":
            dst.vecArray[iMin:iMax,jMin:jMax] = \
                src.vecArray[iMin:iMax,jMin:jMax] - src.vecArray[iMin-1:iMax-1,jMin:jMax]
            dst.local_to_local()
        elif src.location=="left" and dst.location=="element":
            dst.vecArray[iMin:iMax,jMin:jMax] = \
                src.vecArray[iMin+1:iMax+1,jMin:jMax] - src.vecArray[iMin:iMax,jMin:jMax]
            dst.local_to_local()
        else:
            message = "Error: location of dst is not suitable for src.\n"\
                      + f"src_loc={src.location}\n"\
                      + f"dst_loc={dst.location}"
            raise Exception(message)
    def set_diffy(self, src, boundary_type="none"):
        dst = self
        if boundary_type != "none":
            src.set_boundary(boundary_type=boundary_type)
        (iMin, iMax), (jMin, jMax) = dst.da.getRanges()
        nx, ny = dst.da.getSizes()
        if src.location=="element" and dst.location=="down":
            dst.vecArray[iMin:iMax,jMin:jMax] = \
                src.vecArray[iMin:iMax,jMin:jMax] - src.vecArray[iMin:iMax,jMin-1:jMax-1]
            dst.local_to_local()
        elif src.location=="down" and dst.location=="element":
            dst.vecArray[iMin:iMax,jMin:jMax] = \
                src.vecArray[iMin:iMax,jMin+1:jMax+1] - src.vecArray[iMin:iMax,jMin:jMax]
            dst.local_to_local()
        else:
            message = "Error: location of dst is not suitable for src.\n"\
                      + f"src_loc={src.location}\n"\
                      + f"dst_loc={dst.location}"
            raise Exception(message)
    def get_slice(self, edge=False):
        (i0, i1), (j0, j1)  = self.da.getRanges()
        (iMax, jMax) = self.da.getSizes()
        if not edge:
            if self.location=="left":
                i0 = np.maximum(i0, 1)
                i1 = np.minimum(i1, iMax-1)
            if self.location=="down":
                j0 = np.maximum(j0, 1)
                j1 = np.minimum(j1, jMax-1)
        return (SliceEx(i0,i1), SliceEx(j0,j1))
    def get_values(self, ij_list):
        values = np.zeros(len(ij_list))
        for k, (i,j) in enumerate(ij_list):
            values[k] = self.vecArray[i,j]
        return np.array(values)
class DMDAStencil(DMDAWrapper):
    def setup(self, variable_src, stencil_width):
        self.shape = variable_src.da.getSizes()
        self.da = PETSc.DMDA().create(dim = 2, \
            boundary_type=('ghosted', 'ghosted'), \
            sizes = self.shape, \
            dof = 1, stencil_width = stencil_width, \
            stencil_type = 'box', comm = self.comm)
        self.naturalVec = self.da.createNaturalVec()
        self.globalVec = self.da.createGlobalVec()
        self.localVec = self.da.createLocalVec()
        self.vecArray = self.da.getVecArray(self.localVec)
        self.scat, self.rank0 = PETSc.Scatter.toZero(self.naturalVec)
    def copy_from(self, variable_src):
        i, j = variable_src.get_slice()
        self.vecArray[i[0],j[0]] = variable_src.vecArray[i[0],j[0]]
        self.local_to_local()

class DMStagDA(DMDAWrapper):
    def setup(self, dmstag):
        self.da, self.davec = dmstag.dm.VecSplitToDMDA(dmstag.vg,
                                                    self.location,
                                                    self.dof_index)
        if not self.local:
            self.naturalVec = self.da.createNaturalVec()
            self.globalVec = self.da.createGlobalVec()
            self.scat, self.rank0 = PETSc.Scatter.toZero(self.naturalVec)
        self.localVec = self.da.createLocalVec()
        self.vecArray = self.da.getVecArray(self.localVec)

class DMDAHierarchy(DMDAWrapper):
    def setup(self, shape_base, nlevel):
        self.shape_base = shape_base
        self.shape = (shape_base[0] + 1, shape_base[1] + 1)
        self.nlevel = nlevel
        self.da = PETSc.DMDA().create(dim = 2, \
            boundary_type=('ghosted', 'ghosted'), \
            sizes = self.shape, \
            dof = 1, stencil_width = 2, \
            stencil_type = 'box', comm = self.comm)
        self.naturalVec = self.da.createNaturalVec()
        self.globalVec = self.da.createGlobalVec()
        self.localVec = self.da.createLocalVec()
        self.vecArray = self.da.getVecArray(self.localVec)
        self.scat, self.rank0 = PETSc.Scatter.toZero(self.naturalVec)
        if self.nlevel>0:
            self.levels = self.da.coarsenHierarchy(self.nlevel)
            self.mats = []
            self.vecs = []
            mat, vec = self.levels[0].createInterpolation(self.da)
            self.mats.append(mat)
            self.vecs.append(vec)
            for i in range(1, self.nlevel):
                mat, vec = self.levels[i].createInterpolation(self.levels[i-1])
                self.mats.append(mat)
                self.vecs.append(vec)
    def get_coarse(self, dmstagda):
        i, j = self.get_slice()
        dmstagda.local_to_local()
        self.vecArray[i[0],j[0]] = dmstagda.vecArray[i[0],j[0]]
        self.local_to_local()
        self.local_to_global()
        
        matmul = True
        # level 0 to 1
        if matmul:
            self.mats[0].multTranspose(self.globalVec, self.vecs[0])
            self.vecs[0].scale(0.25) # scaling only in coarsening
            da1 = self.levels[0]
            (iMin1, iMax1), (jMin1, jMax1)  = da1.getRanges()
            vl1 = da1.createLocalVec()
            da1.globalToLocal(self.vecs[0], vl1)
            vla1 = da1.getVecArray(vl1)
        else:
            da0 = self.da
            da1 = self.levels[0]
            (iMin1, iMax1), (jMin1, jMax1)  = da1.getRanges()
            iMin0, iMax0 = iMin1*2, iMax1*2
            jMin0, jMax0 = jMin1*2, jMax1*2
            vl0 = da0.createLocalVec()
            vl1 = da1.createLocalVec()
            da0.globalToLocal(self.globalVec, vl0)
            da0.localToLocal(vl0, vl0)
            vla0 = da0.getVecArray(vl0)
            vla1 = da1.getVecArray(vl1)
            vla1[iMin1:iMax1,jMin1:jMax1] = vla0[iMin0:iMax0:2,jMin0:jMax0:2] / 4
            xx = vla0[iMin0-1:iMax0-1:2,jMin0:jMax0:2] + vla0[iMin0+1:iMax0+1:2,jMin0:jMax0:2]
            yy = vla0[iMin0:iMax0:2,jMin0-1:jMax0-1:2] + vla0[iMin0:iMax0:2,jMin0+1:jMax0+1:2]
            xy = vla0[iMin0-1:iMax0-1:2,jMin0-1:jMax0-1:2] + vla0[iMin0+1:iMax0+1:2,jMin0+1:jMax0+1:2]
            yx = vla0[iMin0+1:iMax0+1:2,jMin0-1:jMax0-1:2] + vla0[iMin0-1:iMax0-1:2,jMin0+1:jMax0+1:2]
            vla1[iMin1:iMax1,jMin1:jMax1] += (xx + yy) / 8 + (xy + yx) / 16
            da1.localToGlobal(vl1, self.vecs[0])
        for i in range(1, self.nlevel): 
            if matmul:
                self.mats[i].multTranspose(self.vecs[i-1], self.vecs[i]) # coarsen
                self.vecs[i].scale(0.25) # scaling only in coarsening
            else:
                da0 = self.levels[i-1]
                da1 = self.levels[i]

                (iMin0, iMax0), (jMin0, jMax0)  = da0.getRanges()
                (iMin1, iMax1), (jMin1, jMax1)  = da1.getRanges()
                print("get_coarse ij", self.rank, da0.getRanges(), da1.getRanges(), flush=True)

                iMin0, iMax0 = iMin1*2, iMax1*2
                jMin0, jMax0 = jMin1*2, jMax1*2
                vl0 = da0.createLocalVec()
                vl1 = da1.createLocalVec()
                da0.globalToLocal(self.vecs[i-1], vl0)
                vla0 = da0.getVecArray(vl0)
                vla1 = da1.getVecArray(vl1)
                vla1[iMin1:iMax1,jMin1:jMax1] = vla0[iMin0:iMax0:2,jMin0:jMax0:2] / 4
                xx = vla0[iMin0-1:iMax0-1:2,jMin0:jMax0:2] + vla0[iMin0+1:iMax0+1:2,jMin0:jMax0:2]
                yy = vla0[iMin0:iMax0:2,jMin0-1:jMax0-1:2] + vla0[iMin0:iMax0:2,jMin0+1:jMax0+1:2]
                xy = vla0[iMin0-1:iMax0-1:2,jMin0-1:jMax0-1:2] + vla0[iMin0+1:iMax0+1:2,jMin0+1:jMax0+1:2]
                yx = vla0[iMin0+1:iMax0+1:2,jMin0-1:jMax0-1:2] + vla0[iMin0-1:iMax0-1:2,jMin0+1:jMax0+1:2]
                vla1[iMin1:iMax1,jMin1:jMax1] += (xx + yy) / 8 + (xy + yx) / 16
                da1.localToGlobal(vl1, self.vecs[i])
        cda = self.levels[-1]
        h_coarse_natural = cda.createNaturalVec()
        cda.globalToNatural(self.vecs[-1], h_coarse_natural)
        coarse_scatter, h_coarse_rank0 = PETSc.Scatter.toZero(h_coarse_natural)  
        coarse_scatter.scatter(h_coarse_natural, h_coarse_rank0, False, PETSc.Scatter.Mode.FORWARD) 
        if self.rank==0:
            (NX,NY) = self.levels[-1].sizes
            h_coarse_wide = np.zeros(NY*NX, dtype=np.float64)
            h_coarse = np.zeros((NY-1, NX-1), dtype=np.float64)
            h_coarse_wide[:] = h_coarse_rank0
            h_coarse[:] = h_coarse_wide.reshape(NY,NX)[:NY-1, :NX-1]
        else:
            h_coarse = None
        return h_coarse
    
    def interp2d(self, z0, nx1, ny1):
        ny0, nx0 = z0.shape
        if ny1==ny0:
                z_ = z0
        else:
            w = np.linspace(0,1,ny1).reshape(-1,1)
            if ny1==ny0-1:
                z_ = (1-w) * z0[:-1,:] + w * z0[1:,:]
            elif ny1==ny0+1:
                z_ = np.zeros((ny1, nx0))
                z_[:-1,:] += (1-w[:-1]) * z0
                z_[1: ,:] += w[1:]      * z0
            else:
                raise ValueError(f"Error: |ny1-ny0|>1: ny1={ny1}, ny0={ny0}")
        if nx1==nx0:
            z1 = z_
        else:
            w = np.linspace(0,1,nx1).reshape(1,-1)
            if nx1==nx0-1:
                z1 = (1-w) * z_[:,:-1] + w * z_[:,1:]
            elif nx1==nx0+1:
                z1 = np.zeros((ny1, nx1))
                z1[:,:-1] += (1-w[:,:-1]) * z_
                z1[:,1: ] += w[:,1:]      * z_
            else:
                raise ValueError(f"Error: |nx1-nx0|>1: nx1={nx1}, nx0={nx0}")
        return z1

    def to_fine(self, z_coarse, dmstagda):
        cda = self.levels[-1]
        z_coarse_natural = cda.createNaturalVec()
        coarse_scatter, z_coarse_rank0 = PETSc.Scatter.toZero(z_coarse_natural) 
        # print(self.rank, flush=True)
        if self.rank==0:
            nx1, ny1 = self.levels[-1].sizes
            z_coarse_rank0[:] = self.interp2d(z_coarse, nx1, ny1)

        coarse_scatter.scatter(z_coarse_rank0, z_coarse_natural, False, PETSc.Scatter.Mode.REVERSE)
        cda.naturalToGlobal(z_coarse_natural, self.vecs[-1])
        for i in range(self.nlevel-1, 0, -1):
            self.mats[i].mult(self.vecs[i], self.vecs[i-1]) # refine
        self.mats[0].mult(self.vecs[0], self.globalVec) # refine
        self.da.globalToLocal(self.globalVec, self.localVec)
        self.da.localToLocal(self.localVec, self.localVec)
        i, j = dmstagda.get_slice()
        dmstagda.vecArray[i[0],j[0]] = self.vecArray[i[0],j[0]]
        dmstagda.local_to_local()
class DMStagVariableSet(object):
    def __init__(self):
        self.variables=[]
        self.location=[]
        self.dof_index=[]
        self.dofs = [0,0,0]
    def add_element(self, dmda_element):
        dmda_element.location = "element"
        dmda_element.dof_index = self.location.count("element")
        self.location.append("element")
        self.dofs[2] += 1
        self.variables.append(dmda_element)
    def add_left_and_down(self, dmda_left, dmda_down):
        dmda_left.location = "left"
        dmda_down.location = "down"
        count = self.location.count("left")
        dmda_left.dof_index = count
        dmda_down.dof_index = count
        self.location.append("left")
        self.location.append("down")
        self.dofs[1] += 1
        self.variables.append(dmda_left)
        self.variables.append(dmda_down)
    def add_downleft(self, dmda_downleft):
        dmda_downleft.location = "down_left"
        count = self.location.count("down_left")
        dmda_downleft.dof_index = count
        self.location.append("down_left")
        self.dofs[0] += 1
        self.variables.append(dmda_downleft)
class DMStagBase(object):
    STENCIL = DMStag.StencilType.BOX
    DIM = 2
    BOUNDARY = (PETSc.DM.BoundaryType.GHOSTED,) * DIM

    def __init__(self, dmda_variables, shape, range):
        self.dm = PETSc.DMStag().create(dim=self.DIM)
        self.dm.setStencilType(DMStag.StencilType.BOX)
        self.dm.setStencilWidth(1)
        self.dm.setBoundaryTypes(self.BOUNDARY)
        self.dm.setDof(dmda_variables.dofs)
        self.dm.setGlobalSizes(shape)
        self.dm.setUp()
        self.dm.setCoordinateDMType('stag')
        self.dm.setUniformCoordinates(*range)
        self.vg = self.dm.createGlobalVec()


