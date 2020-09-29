cdef class PSF:
    cdef double[:] get_values(self, double x, double y, int sipm_id) nogil
    cpdef double[:] get_z_bins(self)
    cdef int[:] get_list_of_sipms(self, double x, double y) nogil