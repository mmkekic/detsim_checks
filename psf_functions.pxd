cdef class LT:
    cdef double* get_values_(self, const double x, const double y, int sensor_id) nogil
    cdef double[:] get_zbins_(self)
    cpdef int[:] get_sensor_ids(self)
