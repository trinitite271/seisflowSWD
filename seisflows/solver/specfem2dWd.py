#!/usr/bin/env python3
"""
This class provides utilities for the Seisflows solver interactions with
Specfem2D. It builds upon the base Specfem class which generalizes all solver
interactions with various versions of Specfem.

TODO
    Internal paramater f0 is not currently used. Can we remove or integrate?
"""
import sys
import os
from glob import glob
import pathlib
from seisflows.solver.specfem2d import Specfem2D
from seisflows.tools import unix
import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import spdiags
from concurrent.futures import ProcessPoolExecutor, wait
from seisflows import logger


class Specfem2Dwd(Specfem2D):
    """
    Solver SPECFEM2D
    ----------------
    SPECFEM2D-specific alterations to the base SPECFEM module

    Parameters
    ----------
    :type source_prefix: str
    :param source_prefix: Prefix of source files in path SPECFEM_DATA. Defaults
        to 'SOURCE'
    :type multiples: bool
    :param multiples: set an absorbing top-boundary condition

    Paths
    -----
    ***
    """
    __doc__ = Specfem2D.__doc__ + __doc__

    def __init__(self, **kwargs):
        """Instantiate a Specfem2D solver interface"""
        super().__init__(**kwargs)
        self.path['dispersion_pre'] = os.path.join(self.path.scratch, "dispersion")

    def setup(self):
        super().setup()
        _required_structure = {"disp/obs", "disp/syn", "disp/adj"}
        source_paths = glob(self.path.scratch + "/*")
        self.initialize_disp_directory(_required_structure, source_paths)

    def read_shot(self,path_shot):
        path_shot1 = pathlib.Path(path_shot)
        traces = sorted(path_shot1.glob("*BXZ.semd"))
        shot_data = []
        for trace in traces:
            shot_data.append(np.loadtxt(trace)[:, 1])

        shot_data = np.array(shot_data)
        return shot_data.T
    

    def RTpr(self,seismo_v1,ii,df,dt,vmin,vmax,fmin,fmax,ng,dg,offset,M):
        np1 = vmax-vmin+1
        m=0
        ii = ii+1
        nend=m+offset+M*ii-M
        if nend>ng:
            nend=ng
        uxt=seismo_v1[:,(M*ii-M)+m:nend].T
        offset=nend-(M*ii-M+1+m)+1
        # print(nend,offset)
        x=m+np.linspace(0,offset-1,offset)*dg
        x = x.reshape(1,-1)
        ccn=int(np.fix(1/df/dt))
        d=np.fft.fft(uxt,n=ccn,axis=1)
        lf = int(np.round(fmin/df)+1)
        nf = int(np.round(fmax/df)+1)
        pp = np.zeros((vmax-vmin + 1,1))
        ll0 = np.zeros((np1,nf))
        pp[:,0] = 1/np.linspace(vmin,vmax,vmax-vmin+1)
        ll0=1j*2*np.pi*df*np.dot(pp,x)

        ll0_lou = np.expand_dims(ll0,0).repeat(nf-lf+1,axis=0)
        luo = np.arange(lf-1,nf)
        luo=np.expand_dims(luo,1)
        luo=np.expand_dims(luo,2)
        ll0_lou = ll0_lou*luo.repeat(np1,axis=1).repeat(offset,axis=2)
        l_3d = np.exp(ll0_lou)
        d1 = d[:,(lf-1):nf].T
        mm = np.sum(l_3d * np.expand_dims(d1,1).repeat(np1,axis=1),axis=2).T
        ml = np.abs(mm)
        uxtposr = np.arange((M*ii-M)+m,nend)
        saveForBackward = {'mm':mm,'ll0':ll0,'lf':lf,'nf':nf,'ccn':ccn,'mlr':ml,'uxtposr':uxtposr}

        return ml,saveForBackward
    
    def RTpr1(self,seismo_v1,ii,df,dt,vmin,vmax,fmin,fmax,ng,dg,offset,M):
        np1 = vmax-vmin+1
        m=0
        ii = ii+1
        nend=m+offset+M*ii-M
        if nend>ng:
            nend=ng
        uxt=seismo_v1[:,(M*ii-M)+m:nend].T
        offset=nend-(M*ii-M+1+m)+1
        # print(nend,offset)
        x=m+np.linspace(0,offset-1,offset)*dg
        x = x.reshape(1,-1)
        ccn=int(np.fix(1/df/dt))
        d=np.fft.fft(uxt,n=ccn,axis=1)
        lf = int(np.round(fmin/df)+1)
        nf = int(np.round(fmax/df)+1)
        pp = np.zeros((vmax-vmin + 1,1))
        ll0 = np.zeros((np1,nf))
        pp[:,0] = 1/np.linspace(vmin,vmax,vmax-vmin+1)
        ll0=1j*2*np.pi*df*np.dot(pp,x)
        mm=np.zeros((np1,nf),dtype=np.complex64) 
        l=np.zeros((np1,offset),dtype=np.complex64)

        for luoj in range(lf-1,nf):
            l = np.exp(ll0*(luoj))
            mm[:,luoj]=np.dot(l,d[:,luoj])
        
        ml = np.abs(mm)[:,(lf-1):nf]
        return ml


    def RTpl(self,seismo_v1,ii,df,dt,vmin,vmax,fmin,fmax,b,dg,offset,M):
        np1 = vmax-vmin+1
        m=0
        ii = ii+1
        nend=M*ii-M+1-offset-m
        if nend<1:
            nend=0
        uxt=np.fliplr(seismo_v1[:,nend:(M*ii-M)-m+1]).T
        offset=M*ii-M+1-m-nend
        x=m+np.linspace(0,offset-1,offset)*dg
        x = x.reshape(1,-1)
        ccn=int(np.fix(1/df/dt))
        d=np.fft.fft(uxt,n=ccn,axis=1)
        lf = int(np.round(fmin/df)+1)
        nf = int(np.round(fmax/df)+1)
        pp = np.zeros((vmax-vmin + 1,1))
        ll0 = np.zeros((np1,nf))
        pp[:,0] = 1/np.linspace(vmin,vmax,vmax-vmin+1)
        ll0=1j*2*np.pi*df*np.dot(pp,x)

        ll0_lou = np.expand_dims(ll0,0).repeat(nf-lf+1,axis=0)
        luo = np.arange(lf-1,nf)
        luo=np.expand_dims(luo,1)
        luo=np.expand_dims(luo,2)
        ll0_lou = ll0_lou*luo.repeat(np1,axis=1).repeat(offset,axis=2)
        l_3d = np.exp(ll0_lou)
        d1 = d[:,(lf-1):nf].T
        mm = np.sum(l_3d * np.expand_dims(d1,1).repeat(np1,axis=1),axis=2).T
        ml = np.abs(mm)
        uxtposl = np.arange(nend,(M*ii-M)-m+1)
        saveForBackward = {'mm':mm,'ll0':ll0,'lf':lf,'nf':nf,'ccn':ccn,'mll':ml,'uxtposl':uxtposl}
        return ml,saveForBackward

    def extractDispCurve(self, ml):
        method = 3
        if method == 1:# argmax
            ml = norm_trace(ml)
            curve=np.argmax(ml,axis=0)
        elif method == 2:# Simply extracting the dispersion curve from left to right based on the trend, not recommended
            ml = norm_trace(ml)
            curve=self.extractDispCurve_w2(ml)
        elif method == 3:
            index = np.unravel_index(ml.argmax(), ml.shape)
            curve=self.extractDispCurve_w3(norm_trace(ml),index[1])
        return curve
    
    def extractDispCurve_w2(self, ml):
        [_,nnf] = ml.shape
    #     curve = torch.argmax(ml,dim=0)
        interpz = np.zeros(nnf)
        curve = np.zeros(nnf)
        curve[0] = np.argmax(ml[:,0])
        pos_f = 1
        pos_b = 0
        for k in range(1,nnf):
            
            pfpk, _ = find_peaks(ml[:,pos_f])
            if pfpk.size==0:
                pfpk = np.argmax(ml[:,pos_f])
                pfpk = np.array([pfpk])
            minp = np.argmin(np.abs(pfpk-curve[pos_b]))
            pfpk_val, _ = find_peaks(ml[:,pos_b])
            if pfpk_val.size==0:
                pfpk_val = np.argmax(ml[:,pos_b])
                pfpk_val = np.array([pfpk_val])
            minp_val = np.argmin(np.abs(pfpk_val-pfpk[minp]))
            if pfpk_val[minp_val]==curve[pos_b]:
                curve[pos_f] = pfpk[minp]
                pos_f = pos_f + 1
                pos_b = pos_f - 1
            else:
                # break
                pos_f = pos_f + 1
        if np.nonzero(curve)[0].size==0:
            curve = curve
        else:
            curve = np.interp(np.linspace(0,nnf-1,nnf), np.nonzero(curve)[0], curve[np.nonzero(curve)])
        return curve
        
    def extractDispCurve_w3(self, ml,ini):
        [np1,nnf] = ml.shape
        curve = np.zeros(nnf)
        curve_dm = np.zeros(nnf)
        curve_um = np.zeros(nnf)
        
        curve[ini] = np.argmax(ml[:,ini])
        pfpk1, _ = find_peaks(-ml[:,ini])
        pfpk1 = np.append(0,pfpk1)
        pfpk1 = np.append(pfpk1,np1)
    #     print(pfpk1-curve[0],pfpk1,curve[0])
        d_loc = np.argwhere((pfpk1-curve[ini])>0)[0][0]
        curve_um[ini] = pfpk1[d_loc]
        curve_dm[ini] = pfpk1[d_loc-1]
        pos_f = ini+1
        pos_b = ini
        for k in range(ini+1,nnf):
            
            pfpk, _ = find_peaks(ml[:,pos_f])
            if pfpk.size==0:
                pfpk = np.argmax(ml[:,pos_f])
                pfpk = np.array([pfpk])
            minp = np.argmin(np.abs(pfpk-curve[pos_b]))
            pfpk_val, _ = find_peaks(ml[:,pos_b])
            if pfpk_val.size==0:
                pfpk_val = np.argmax(ml[:,pos_b])
                pfpk_val = np.array([pfpk_val])
            minp_val = np.argmin(np.abs(pfpk_val-pfpk[minp]))
            if pfpk_val[minp_val]==curve[pos_b]:
                curve[pos_f] = pfpk[minp]
                # find wideth
                pfpk1, _ = find_peaks(-ml[:,pos_f])
                pfpk1 = np.append(0,pfpk1)
                pfpk1 = np.append(pfpk1,np1)
                d_loc = np.argwhere((pfpk1-curve[pos_f])>0)[0][0]
                curve_um[pos_f] = pfpk1[d_loc]
                curve_dm[pos_f] = pfpk1[d_loc-1]
                pos_f = pos_f + 1
                pos_b = pos_f - 1
            else:
    #             break    
                pos_f = pos_f + 1
        pos_f = ini-1
        pos_b = ini
        for k in range(ini-1,0,-1):
            pfpk, _ = find_peaks(ml[:,pos_f])
            if pfpk.size==0:
                pfpk = np.argmax(ml[:,pos_f])
                pfpk = np.array([pfpk])
            minp = np.argmin(np.abs(pfpk-curve[pos_b]))
            pfpk_val, _ = find_peaks(ml[:,pos_b])
            if pfpk_val.size==0:
                pfpk_val = np.argmax(ml[:,pos_b])
                pfpk_val = np.array([pfpk_val])
            minp_val = np.argmin(np.abs(pfpk_val-pfpk[minp]))
            if pfpk_val[minp_val]==curve[pos_b]:
                curve[pos_f] = pfpk[minp]
                # find wideth
                pfpk1, _ = find_peaks(-ml[:,pos_f])
                pfpk1 = np.append(0,pfpk1)
                pfpk1 = np.append(pfpk1,np1)
                d_loc = np.argwhere((pfpk1-curve[pos_f])>0)[0][0]
                curve_um[pos_f] = pfpk1[d_loc]
                curve_dm[pos_f] = pfpk1[d_loc-1]
                pos_f = pos_f - 1
                pos_b = pos_f + 1
            else:
                # break
                pos_f = pos_f - 1
        if np.nonzero(curve)[0].size==0:
            curve = curve
        else:
            curve = np.interp(np.linspace(0,nnf-1,nnf), np.nonzero(curve)[0], curve[np.nonzero(curve)])
        return curve

    def extractDispCurve_w4(ml,ini):
        ml = ml.cpu().detach().np()
        [np1,nnf] = ml.shape
    #     curve = torch.argmax(ml,dim=0)
    #     interpz = np.zeros(nnf)
        curve = np.zeros(nnf)
        curve_dm = np.zeros(nnf)
        curve_um = np.zeros(nnf)
        
        curve[ini] = np.argmax(ml[:,ini])
        pos_f = ini+1
        pos_b = ini
        for k in range(ini+1,nnf):
            
            pfpk, _ = find_peaks(ml[:,pos_f])
            if pfpk.size==0:
                pfpk = np.argmax(ml[:,pos_f])
                pfpk = np.array([pfpk])
            minp = np.argmin(np.abs(pfpk-curve[pos_b]))
            pfpk_val, _ = find_peaks(ml[:,pos_b])
            if pfpk_val.size==0:
                pfpk_val = np.argmax(ml[:,pos_b])
                pfpk_val = np.array([pfpk_val])
            minp_val = np.argmin(np.abs(pfpk_val-pfpk[minp]))
            if pfpk_val[minp_val]==curve[pos_b]:
                curve[pos_f] = pfpk[minp]
                # find wideth

                pos_f = pos_f + 1
                pos_b = pos_f - 1
            else:
    #             break    
                pos_f = pos_f + 1
        pos_f = ini-1
        pos_b = ini
        for k in range(ini-1,0,-1):
            pfpk, _ = find_peaks(ml[:,pos_f])
            if pfpk.size==0:
                pfpk = np.argmax(ml[:,pos_f])
                pfpk = np.array([pfpk])
            minp = np.argmin(np.abs(pfpk-curve[pos_b]))
            pfpk_val, _ = find_peaks(ml[:,pos_b])
            if pfpk_val.size==0:
                pfpk_val = np.argmax(ml[:,pos_b])
                pfpk_val = np.array([pfpk_val])
            minp_val = np.argmin(np.abs(pfpk_val-pfpk[minp]))
            if pfpk_val[minp_val]==curve[pos_b]:
                curve[pos_f] = pfpk[minp]
                # find wideth
                pos_f = pos_f - 1
                pos_b = pos_f + 1
            else:
                break
        curve =  np.interp(np.linspace(0,nnf-1,nnf), np.nonzero(curve)[0], curve[np.nonzero(curve)],right=0)
        for k in range(nnf):
            pfpk1, _ = find_peaks(-ml[:,k])
            pfpk1 = np.append(0,pfpk1)
            pfpk1 = np.append(pfpk1,np1)
            d_loc = np.argwhere((pfpk1-np.round(curve[k]))>0)[0][0]
            curve_um[k] = pfpk1[d_loc]
            curve_dm[k] = pfpk1[d_loc-1]
        return curve


    def initialize_disp_directory(self, _required_structure, source_paths, max_workers=None):
        """
        Serial or parallel task used to initialize working directories for
        each of the available sources

        :type max_workers: int
        :param max_workers: number of concurrent tasks to use when creating 
            working directories. Defaults to using all available cores on 
            the machine since this is a lightweight task
        """
        if max_workers is None:
            max_workers = unix.nproc() - 1  # use all available cores

        # Full path each source in the scratch directory for directories that
        # do not exist, otherwise this function gets skipped
        
        source_paths = [f for f in source_paths if not f.endswith("mainsolver")]
        if source_paths:
            logger.info(f"adding {self.ntask} solver dispersion directories")
        else:
            return

        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._initialize_disp_directory, cwd, _required_structure)
                    for cwd in source_paths
                ]
            wait(futures)
            # If any of the jobs, calling the result will raise the Exception
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.critical(f"directory initialization error: {e}")
                    sys.exit(-1)
        else:
            for source_name in self.source_names:
                cwd = os.path.join(self.path.scratch, source_name)
                # if os.path.exists(cwd):
                #     continue
                self._initialize_disp_directory(cwd, _required_structure)


    def _initialize_disp_directory(self, cwd=None, _required_structure=None):
        """
        Adding dispersion curve files
        """
        # Define a constant list of required SPECFEM dir structure, relative cwd

        # Allow this function to be called on system or in serial
        if cwd is None:
            cwd = self.cwd
            source_name = self.source_name
        else:
            source_name = os.path.basename(cwd)


        for dir_ in _required_structure:
            unix.mkdir(os.path.join(cwd, dir_))



def norm_trace(seis):
    data_out = np.zeros(np.shape(seis))
    for k in range(np.size(seis,axis=1)):
        data_out[:,k] = seis[:,k]/np.max(np.abs(seis[:,k]))
    seis=data_out
    return seis

def smooth2a(matrixIn,Nr,Nc):
    [row,col] = matrixIn.shape
    eL = spdiags(np.ones((2*Nr,row)),np.arange(-Nr,Nr),row,row)
    eR = spdiags(np.ones((2*Nc,col)),np.arange(-Nc,Nc),col,col)
    nrmlize = eL@(np.ones_like(matrixIn))@eR
    matrixOut = eL@matrixIn@eR
    matrixOut = matrixOut/nrmlize
    return matrixOut