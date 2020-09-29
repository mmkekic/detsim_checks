
import pandas as pd
from invisible_cities.core import system_of_units as units


import numpy as np
cimport numpy as np
from libc.math cimport sqrt, round, ceil, floor
cimport cython



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef int[:] get_list_of_sipms(double x, double y, double minx, double miny, double sipm_pitch, int[:,:,:] nearby_list) nogil:
        cdef int indx_xi, indx_yi
        #still not out-of-range safe
        indx_xi = <int> floor((x-minx)/sipm_pitch)
        indx_yi = <int> floor((y-miny)/sipm_pitch)
        return nearby_list[indx_xi, indx_yi]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef int get_values(double x, double y, int sipm_id, double[:, :] psf_values, double[:] ref_psf_values, double[:] xsipms, double[:] ysipms, double max_psf, double psf_bin) nogil:
    cdef:
        double dist
        int psf_bin_indx
        double xsipm, ysipm, tmp_x, tmp_y
    xsipm = xsipms[sipm_id]
    ysipm = ysipms[sipm_id]
    tmp_x = x-xsipm; tmp_y = y-ysipm
    dist = sqrt(tmp_x*tmp_x + tmp_y*tmp_y)
    if dist>max_psf:
        return 0
    psf_bin_indx = <int> floor(dist/psf_bin)
    ref_psf_values[:] = psf_values[psf_bin_indx,:] 
    return   1

@cython.boundscheck(False)
@cython.wraparound(False)
def electron_loop(np.ndarray[double, ndim=1] xs,
                  np.ndarray[double, ndim=1] ys,
                  double [:] ts,
                  np.ndarray[unsigned long, ndim=1] phs,
                  sipm_database,
                  psf_fname,
                  double EL_drift_velocity,
                  double sipm_time_bin,
                  int num_bins):

    cdef:
        double [:, :] psf_values               
        double [:] xsipms, ysipms, z_bins, sipm_values, xgrid, ygrid
        int [:] z_bins_indcs
        double psf_bin, dz_bin, max_zel, max_psf, EL_z, minx, miny, sipm_pitch
        
    PSF      = pd.read_hdf(psf_fname, "/LightTable")
    Config   = pd.read_hdf(psf_fname, "/Config")
    EL_z     = float(float(Config.loc["EL_GAP"].value) * units.mm)
    el_pitch = float(Config.loc["pitch_z"].value) * units.mm

    z_bins = np.arange(0, EL_z+np.finfo(float).eps, el_pitch).astype(np.double)
    z_bins_indcs = np.arange(len(z_bins)).astype(np.intc)

    sipm_values = np.zeros(len(z_bins), dtype=np.double)
    psf_values = PSF.values/len(z_bins)
    psf_bin    = float(PSF.index[1]-PSF.index[0])
    dz_bin = el_pitch

    xsipms = sipm_database.X.values.astype(np.double)
    ysipms = sipm_database.Y.values.astype(np.double)
    max_zel = EL_z
    max_psf = max(PSF.index.values)
    
    xgrid = np.sort(np.unique(xsipms))
    ygrid = np.sort(np.unique(xsipms))
    minx = min(xgrid)
    miny = min(ygrid)
        
    sipm_pitch = 15.5



 
    max_num = (2*(<int> ceil(max_psf/sipm_pitch)+1))**2
    nearby_list_ = (-1)*np.ones((len(xgrid), len(ygrid), max_num), dtype=np.intc)
    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            msk1 = np.sqrt((np.asarray(xsipms)-np.asarray(xgrid[i]))**2 + (np.asarray(ysipms)-np.asarray(ygrid[j]))**2)<=max_psf+1.5*sipm_pitch
            closest_sipms = np.argwhere(msk1).squeeze()
            nearby_list_[i,j, :len(closest_sipms)]=closest_sipms
    cdef int [:, :, :] nearby_list = nearby_list_

    cdef:

        int nsipms = len(xsipms)
        int [:] sipm_ids = np.empty_like(nearby_list_[0,0])
        double [:] zs = z_bins
        double [:, :] sipmwfs = np.zeros([nsipms, num_bins], dtype=np.double)        
        int indx_sipm
        int indx_el
        int indx_z
        double signal
        double time
        int indx_time
        double [:] EL_times_
        int sipm_id
        double x_el, y_el, ph_el, time_el
    #lets create vector of EL_times
    num_zs = np.copy(zs)
    zs_bs = num_zs[1]-num_zs[0]
    EL_times = (num_zs+zs_bs/2.)/EL_drift_velocity
    EL_times_ = EL_times.astype(np.double)

    cdef double[:] psf_factors_ = np.zeros_like(EL_times_)
    cdef double[:] psf_factors = np.zeros_like(EL_times_)
    cdef int[:] indxs_time = np.zeros_like(EL_times_, dtype=np.intc)
    #cdef int num_0s = 0
    #cdef int tmp_res
    #with nogil:
    #sipm_ids = np.arange(nsipms, dtype=np.intc)
    for indx_el in range(ts.shape[0]):
        x_el = xs[indx_el]
        y_el = ys[indx_el]
        ph_el = phs[indx_el]
        time_el = ts[indx_el]
        
        sipm_ids[:] = get_list_of_sipms(x_el, y_el, minx, miny, sipm_pitch, nearby_list)
        for indx_sipm in range(sipm_ids.shape[0]):
            sipm_id = sipm_ids[indx_sipm]
            if sipm_id>-1:
                tmp_res = get_values(x_el, y_el, sipm_id, psf_values, psf_factors, xsipms, ysipms, max_psf, psf_bin)
                if tmp_res==0:
                    continue
                else:
                    for indx_z in range(zs.shape[0]):
                        time = time_el+EL_times_[indx_z]
                        indxs_time[indx_z] = <int> floor(time/sipm_time_bin)        
                        #if indx_time>=num_bins:
                        #    continue
                        signal = psf_factors[indx_z] * ph_el
                        sipmwfs[sipm_id, indxs_time[indx_z]] += signal
                    
            else:
                break
    #print (num_0s)
    return sipmwfs
