cdef class PSF:
    cdef int get_values(self, const double x, const double y, const int sipm_id, double [:] psf_values) nogil
    cpdef double[:] get_z_bins(self)
    cdef int[:] get_list_of_sipms(self, const double x, const double y) nogil