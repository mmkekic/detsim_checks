cdef class PSF:
    cdef double* get_values(self, const double x, const double y, int sipm_id)
    cpdef double[:] get_z_bins(self)
    cpdef int[:] get_sipm_ids(self)