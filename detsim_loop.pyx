
cimport psf_functions
from psf_functions cimport PSF
import numpy as np
import pandas as pd

cimport numpy as np
from libc.math cimport sqrt, round, ceil, floor
cimport cython

#@profile
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def electron_loop(double [:] xs,
                  double [:] ys,
                  double [:] ts,
                  np.ndarray[unsigned long, ndim=1] phs,
                  PSF PSF,
                  double EL_drift_velocity,
                  double sipm_time_bin,
                  int num_bins):
                  
    cdef:
        int [:] sipm_ids = np.empty_like(PSF.nearby_list[0,0])
        int nsipms = PSF.sipm_ids.shape[0]
        double [:] zs = PSF.get_z_bins()
        double [:, :] sipmwfs = np.zeros([nsipms, num_bins], dtype=np.double)
        int important
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
    EL_times = (num_zs+zs_bs/2.)/EL_drift_velocity/sipm_time_bin
    EL_times_ = EL_times.astype(np.double)

    cdef double[:] psf_factors = np.empty_like(EL_times_, dtype=np.double)
    #cdef double[:] psf_factors = np.empty(len(EL_times_), dtype=np.double)
    cdef int[:] indxs_time = np.zeros_like(EL_times_, dtype=np.intc)
    cdef int indxs_time_
    with nogil:

        for indx_el in range(ts.shape[0]):
            x_el = xs[indx_el]
            y_el = ys[indx_el]
            ph_el = phs[indx_el]
            time_el = ts[indx_el]/sipm_time_bin
            for sipm_id in range(nsipms):
                important = PSF.get_values(x_el, y_el, sipm_id, psf_factors)
                if important>0:
                    for indx_z in range(zs.shape[0]):
                        time = time_el+EL_times_[indx_z]
                        indxs_time_ = <int> floor(time)
                        
                        #if indx_time>=num_bins:
                        #    continue
                        signal = psf_factors[indx_z] * ph_el
                        sipmwfs[sipm_id, indxs_time_] += signal
    return sipmwfs
