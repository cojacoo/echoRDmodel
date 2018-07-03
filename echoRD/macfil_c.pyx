import numpy as np
cimport cython

cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def macfil_c(object[DTYPE_t, ndim=1] pmzid,object[DTYPE_t, ndim=1] pid, int lpx, int lmx):
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] mfil = np.zeros(lmx, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] floc = np.zeros(lpx, dtype=DTYPE)

    pmzid[pmzid<0]=0
    for i in pid:
        mfil[pmzid[i]] += 1
        floc[i] = mfil[pmzid[i]]

    return [mfil, floc]