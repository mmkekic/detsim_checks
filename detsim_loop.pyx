
cimport psf_functions
from psf_functions cimport LT
import numpy as np
import pandas as pd

cimport numpy as np
from libc.math cimport sqrt, round, ceil, floor
cimport cython


cdef double[:] spread_histogram(const double[:] histogram, int nsmear_left, int nsmear_right):
    cdef int nsamples = nsmear_left + nsmear_right
    cdef int l = len(histogram)
    cdef double [:] spreaded = np.zeros(l + nsamples-1)
    cdef int h, i, aux
    cdef double v

    for h in range(nsmear_left, l):
        v = histogram[h]/nsamples
        aux = h-nsmear_left
        for i in range(nsamples):
            spreaded[aux + i] += v
    return spreaded[:l]

#@profile
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def electron_loop(double [:] xs,
                  double [:] ys,
                  double [:] ts,
                  np.ndarray[unsigned long, ndim=1] phs,
                  LT LT,
                  double EL_drift_velocity,
                  double EL_gap,
                  double sensor_time_bin,
                  double buffer_length):


    cdef:
        int [:] sns_ids = LT.get_sensor_ids()
        int nsens = sns_ids.shape[0]
        double [:] zs = LT.get_zbins_()
        int num_bins = <int> ceil (buffer_length/sensor_time_bin)
        double [:, :] wfs = np.zeros([nsens, num_bins], dtype=np.double)




    #lets create vector of EL_times
    if len(zs)>1:
        zs_bs = zs[1]-zs[0]
    else:
        zs_bs = EL_gap
    time_bs_sns   = zs_bs/EL_drift_velocity/sensor_time_bin
    max_time_sns  = EL_gap/EL_drift_velocity/sensor_time_bin
    #EL_times_ corresponding to LT z partitions
    cdef double [:] EL_times_ = np.arange(time_bs_sns/2., max_time_sns+np.finfo(np.double).eps, time_bs_sns).astype(np.double)

    #smearing factor in case time_bs_sns is larger than sensor_time_bin
    cdef int nsmear = <int> ceil(time_bs_sns)


    cdef double[::1] LT_factors = np.empty_like(EL_times_, dtype=np.double)
    cdef double * LT_factors_ = &LT_factors[0]


    cdef int indx_el, indx_time, sns_id_indx, sns_id, indx_EL
    cdef double x_el, y_el, ph_el, time_el
    cdef double signal, time

    for indx_el in range(ts.shape[0]):
        x_el = xs[indx_el]
        y_el = ys[indx_el]
        ph_el = phs[indx_el]
        time_el = ts[indx_el]/sensor_time_bin
        for sns_id_indx in range(nsens):
            sns_id = sns_ids[sns_id_indx]
            LT_factors_ = LT.get_values_(x_el, y_el, sns_id)
            if LT_factors_ != NULL:
                for indx_EL in range(EL_times_.shape[0]):
                    time = time_el + EL_times_[indx_EL]
                    indx_time = <int> floor(time)
                    # if indx_time>=num_bins:
                    #     continue
                    signal = LT_factors_[indx_EL] * ph_el
                    
                    wfs[sns_id_indx, indx_time] += signal

    if nsmear>1:
        nsmear_left = <int> (nsmear/2)
        nsmear_right = nsmear - nsmear_left
        for sns_id_indx in range(nsens):
            wfs[sns_id_indx] = spread_histogram(wfs[sns_id_indx], nsmear_left, nsmear_right)
    return wfs


