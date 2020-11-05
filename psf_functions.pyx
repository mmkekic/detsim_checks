import numpy as np
import pandas as pd
from invisible_cities.core import system_of_units as units

cimport numpy as np
from libc.math cimport sqrt, round, ceil, floor
cimport cython
cimport cython

from IPython.core.debugger import set_trace

#base class for PSF
cdef class LT:
    cdef double* get_values_(self, const double x, const double y, const int sensor_id) nogil:
        pass
    cdef double[:] get_zbins_(self):
        pass
    cpdef int[:] get_sensor_ids(self):
        pass

cdef class PSF_distance(LT):
    cdef readonly:
        object PSF
        double [:] xsipms, ysipms, zbins
    cdef:
        double [:, ::1] psf_values
        int [:] sipm_ids
        double psf_bin, dz_bin, max_zel, max_psf, EL_z
        int [:, :, :] nearby_list
        double inv_bin,  inv_pitch, max_psf_sq

    def __init__(self, sipm_database, psf_fname, EL_z=None):
        PSF      = pd.read_hdf(psf_fname, "/LightTable")
        self.PSF = PSF
        Config   = pd.read_hdf(psf_fname, "/Config")
        EL_z     = float(float(Config.loc["EL_GAP"].value) * units.mm)
                
        self.EL_z = EL_z
        el_pitch = float(Config.loc["pitch_z"].value) * units.mm

        self.zbins = np.arange(0, EL_z+np.finfo(float).eps, el_pitch).astype(np.double)

        self.psf_values = np.array(PSF.values/len(self.zbins), order='C', dtype=np.double)
        self.psf_bin    = float(PSF.index[1]-PSF.index[0])
        self.inv_bin = 1./self.psf_bin
        self.dz_bin = el_pitch

        self.xsipms = sipm_database.X.values.astype(np.double)
        self.ysipms = sipm_database.Y.values.astype(np.double)
        self.sipm_ids = sipm_database.index.values.astype(np.intc)
        self.max_zel = EL_z
        self.max_psf = max(PSF.index.values)
        self.max_psf_sq = self.max_psf*self.max_psf

        
    cdef double [:] get_zbins_(self):
        return np.asarray(self.zbins)

    cpdef int [:] get_sensor_ids(self):
        return np.asarray(self.sipm_ids)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef double* get_values_(self, const double x, const double y, const int sipm_id) nogil:
        cdef:
            double dist, aux
            unsigned int psf_bin_id
            double xsipm, ysipm, tmp_x, tmp_y
            double*  psf_values
        xsipm = self.xsipms[sipm_id]
        ysipm = self.ysipms[sipm_id]
        tmp_x = x-xsipm; tmp_y = y-ysipm
        dist = tmp_x*tmp_x + tmp_y*tmp_y
        if dist>self.max_psf_sq:
            return NULL
        aux = sqrt(dist)*self.inv_bin
        psf_bin_id = <int> floor(aux)
        psf_values = &self.psf_values[psf_bin_id, 0]#psf_values_
        return psf_values

    def get_values(self, const double x, const double y, const int sipm_id):
        """ This is only for using within python"""
        cdef double* pointer
        pointer=self.get_values_(x, y, sipm_id)
        numpy_array = np.asarray(<np.double_t[:self.zbins.shape[0]]> pointer)
        return numpy_array

import bisect
from invisible_cities.cities.detsim_get_psf import binedges_from_bincenters


cdef class LTS2(LT):
    #cdef double [:, ::1] psf_values
    cdef readonly:
        object lt
        double [:]  zbins
    cdef:
        double [:, :, :, ::1] tensor
        double [:] xcenters, ycenters, xbins, ybins
        int [:] pmt_ids
        double LT_bin,  max_zel, EL_z, r_max2, r_max
        double inv_x, inv_y
        int lenz

    def __init__(self, LT_fname, EL_z):
        lt = pd.read_hdf(LT_fname, "LightTable")
        config   = pd.read_hdf(LT_fname, "/Config")
        sensor     = config.loc["sensor"]     .value
        lt = lt.drop(columns = [sensor + "_total"])
        
        self.r_max = (config.loc['ACTIVE_rad'].astype(float).value)
        self.r_max2 = self.r_max**2
        self.zbins = np.array([EL_z/2])
        self.lt = lt
        self.lenz = len(self.zbins)
        self.pmt_ids = np.arange(lt.shape[1]).astype(np.intc)
        xcenters = np.sort(np.unique(lt.index.get_level_values('x')))
        ycenters = np.sort(np.unique(lt.index.get_level_values('y')))
        indx = pd.MultiIndex.from_product([xcenters, ycenters], names=['x', 'y'])
        tensor_aux = self.lt.reindex(indx, fill_value=0).values.reshape(len(xcenters), len(ycenters), lt.shape[1])[..., None]

        self.tensor = np.asarray(np.repeat(tensor_aux, self.lenz, axis=-1), dtype=np.double, order='C')
        self.xcenters = xcenters
        self.ycenters = ycenters
        self.inv_x = 1./(xcenters[1]-xcenters[0])
        self.inv_y = 1./(ycenters[1]-ycenters[0])
        
    cdef double [:] get_zbins_(self):
        return np.asarray(self.zbins)

    cpdef int [:] get_sensor_ids(self):
        return np.asarray(self.pmt_ids)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef double* get_values_(self, const double x, const double y, const int sns_id) nogil:
        cdef:
            double*  values
            int xindx_, yindx_
            double xval, yval, value
        if (x*x+y*y)>=self.r_max2 :
            return NULL
        xindx_ = <int> round((x+self.r_max)*self.inv_x)
        yindx_ = <int> round((y+self.r_max)*self.inv_y)
        values = &self.tensor[xindx_, yindx_, sns_id, 0]
        return values
