import numpy as np
import pandas as pd
from invisible_cities.core import system_of_units as units

cimport numpy as np
from libc.math cimport sqrt, round, ceil, floor
cimport cython
cimport cython

from IPython.core.debugger import set_trace

#base class for PSF
cdef class PSF:
    cdef int get_values(self, const double x, const double y, const int sipm_id, double[:] psf_values) nogil:
        pass
    cpdef double[:] get_z_bins(self):
        pass
    def get_sipm_ids(self):
        pass
    cdef int[:] get_list_of_sipms(self, const double x, const double y) nogil:
        pass

cdef class PSF_distance(PSF):
    cdef double [:, :] psf_values               
    cdef public:
        double [:] xsipms, ysipms, z_bins, sipm_values, xgrid, ygrid
        int [:] sipm_ids, z_bins_indcs
        double psf_bin, dz_bin, max_zel, max_psf, EL_z, minx, miny, sipm_pitch
        int [:, :, :] nearby_list
        double inv_bin,  inv_pitch, max_psf_sq



    def __init__(self, sipm_database, psf_fname, z_bins=None):
        PSF      = pd.read_hdf(psf_fname, "/LightTable")
        Config   = pd.read_hdf(psf_fname, "/Config")
        EL_z     = float(float(Config.loc["EL_GAP"].value) * units.mm)
                
        self.EL_z = EL_z
        el_pitch = float(Config.loc["pitch_z"].value) * units.mm

        self.z_bins = np.arange(0, EL_z+np.finfo(float).eps, el_pitch).astype(np.double)
        self.z_bins_indcs = np.arange(len(self.z_bins)).astype(np.intc)

        self.sipm_values = np.zeros(len(self.z_bins), dtype=np.double)
        self.psf_values = PSF.values/len(self.z_bins)
        self.psf_bin    = float(PSF.index[1]-PSF.index[0])
        self.inv_bin = 1./self.psf_bin
        self.dz_bin = el_pitch

        self.xsipms = sipm_database.X.values.astype(np.double)
        self.ysipms = sipm_database.Y.values.astype(np.double)
        self.sipm_ids = sipm_database.index.values.astype(np.intc)
        self.max_zel = EL_z
        self.max_psf = max(PSF.index.values)
        self.max_psf_sq = self.max_psf*self.max_psf
        cdef size_t dummy_indx
        self.xgrid = np.sort(np.unique(self.xsipms))
        self.ygrid = np.sort(np.unique(self.xsipms))
        self.minx = min(self.xgrid)
        self.miny = min(self.ygrid)
        
        self.sipm_pitch = 15.5
        self.inv_pitch = 1./self.sipm_pitch
        self.nearby_list = self.__get_binned_matrix__()

    def __get_binned_matrix__(self):
        max_num = (2*(<int> ceil(self.max_psf/self.sipm_pitch)+1))**2
        full_list = (-1)*np.ones((len(self.xgrid), len(self.ygrid), max_num), dtype=np.intc)
        for i in range(len(self.xgrid)):
            for j in range(len(self.ygrid)):
                msk1 = np.sqrt((np.asarray(self.xsipms)-np.asarray(self.xgrid[i]))**2 + (np.asarray(self.ysipms)-np.asarray(self.ygrid[j]))**2)<=self.max_psf+1.5*self.sipm_pitch
                closest_sipms = np.argwhere(msk1).ravel()
                full_list[i,j, :len(closest_sipms)]=closest_sipms.flatten()
        return full_list

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef int[:] get_list_of_sipms(self, const double x, const double y) nogil:
        cdef:
            unsigned int indx_xi, indx_yi
            double inter_x, inter_y
        inter_x = (x-self.minx)*self.inv_pitch
        inter_y = (y-self.miny)*self.inv_pitch
        #still not out-of-range safe
        indx_xi = <int> floor(inter_x)
        indx_yi = <int> floor(inter_y)
        return self.nearby_list[indx_xi, indx_yi]
        
    cpdef double [:] get_z_bins(self):
        return np.asarray(self.z_bins)

    def get_sipm_ids(self):
        return np.asarray(self.sipm_ids)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef int get_values(self, const double x, const double y, const int sipm_id, double [:] psf_values) nogil:
        cdef:
            double dist, aux
            unsigned int psf_bin
            double xsipm, ysipm, tmp_x, tmp_y
            #double [:] psf_values_
        xsipm = self.xsipms[sipm_id]
        ysipm = self.ysipms[sipm_id]
        tmp_x = x-xsipm; tmp_y = y-ysipm
        dist = tmp_x*tmp_x + tmp_y*tmp_y
        if dist>self.max_psf_sq:
            return 0
        aux = sqrt(dist)*self.inv_bin
        psf_bin = <int> floor(aux)
        #psf_values_ = self.psf_values[psf_bin,:]
        psf_values[:] = self.psf_values[psf_bin, :]#psf_values_
        return 1
