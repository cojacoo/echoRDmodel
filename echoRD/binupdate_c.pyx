
import numpy as np
cimport cython
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPEf = np.float
ctypedef np.float_t DTYPEf_t

#DTYPEb = np.bool
#ctypedef np.bool_t DTYPEb_t

# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False


from libc.stdlib cimport malloc, free
from cython cimport view
np.import_array()
 
ctypedef np.float64_t FLOAT_t
ctypedef np.intp_t INT_t
ctypedef np.ulong_t INDEX_t
ctypedef np.uint8_t BOOL_t

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct Sorter:
    INT_t index
    INT_t value

cdef int _compare(const_void *a, const_void *b):
    cdef INT_t v = ((<Sorter*>a)).value-((<Sorter*>b)).value
    if v < 0: return -1
    if v >= 0: return 1

cdef void cyargsort(INT_t[:] data, Sorter * order):
    cdef INT_t i
    cdef INT_t n = data.shape[0]
    for i in range(n):
        order[i].index = i
        order[i].value = data[i]
    qsort(<void *> order, n, sizeof(Sorter), _compare)
    
cpdef argsort(INT_t[:] data, INT_t[:] order):
    cdef INT_t i
    cdef INT_t n = data.shape[0]
    cdef Sorter *order_struct = <Sorter *> malloc(n * sizeof(Sorter))
    cyargsort(data, order_struct)
    for i in range(n):
        order[i] = order_struct[i].index
    free(order_struct)

#@cython.boundscheck(False) # turn off bounds-checking for entire function
#@cython.wraparound(False)  # turn off negative index wrapping for entire function

def binupdate_c(object[INT_t, ndim=1] bins, FLOAT_t dt, FLOAT_t ksref, FLOAT_t Dref, INT_t spltf, INT_t lbin):

    cdef np.ndarray[INT_t, ndim=1] order
    cdef np.ndarray[INT_t, ndim=1] LTEref
    cdef np.ndarray[INT_t, ndim=1] deviation
    cdef np.ndarray[INT_t, ndim=1] bins_empty
    cdef FLOAT_t tmix
    cdef FLOAT_t step_theoret

    order = np.zeros(len(bins), dtype=np.intp)
    argsort(bins, order) #sort bins
    LTEref = np.arange(len(bins))  # bin setting at LTE
    deviation = bins[order] - LTEref  # deviation between the two
    # reorganise single step deviations:
    # onestep=np.where(abs(deviation)==1)[0]
    bins[order[abs(deviation) == 1]] -= deviation[abs(deviation) == 1]
    deviation = bins[order] - LTEref  # deviation between the two
    if any(deviation < 0):  # if there are still particles of smaller pores they are shifted to the enc
        # bins[order[deviation<0]]=np.amax(bins)+np.arange(np.sum(deviation<0))
        bins[order[deviation < 0]] = np.intp(LTEref[deviation < 0] - deviation[deviation < 0])
    if any(deviation > 0):  # if there are still particles of larger pores they mix towards LTE with t_mix
        # allow reorganisation to LTE with tmix=((ks*dt)**2)/Dmix
        # after tmix LTE is achieved this compares to dt and gives the current reorganisation maximum.
        # tmix=((mc.soilmatrix.ks[mc.soilgrid.ravel()[cell]-1]*mc.dt)**2)/mc.D[np.intp(np.round(np.amin([len(mc.D)-1,np.percentile(bins,mc.LTEpercentile)]))),mc.soilgrid.ravel()[cell]-1]
        # alternative: tmix draws from D in empty bins (bin percentile to saturation)
        tmix = ((ksref * dt) ** 2) / Dref
        step_theoret = np.amax(deviation) * (dt / spltf) / tmix  # given tmix as time to reach LTE the current deviation from LTE is projected to be reduced within the current time step by its share
        bins[order[deviation > 0]] = np.intp(LTEref[deviation > 0] + np.fmax(deviation[deviation > 0] - step_theoret, 0))

    # this is still an issue but deprecated for now:
    # mc.LTEmemory[cell]+=step_theoret #through the discrete step definition an accumulation is needed. DEBUG: this is a far too simple representation at it is bound to the cell not to the particles
    # DEBUG:
    if any(bins >= lbin):
        bins[bins >= lbin] = np.random.randint(lbin - 1, size=np.sum(bins >= lbin))

    return [bins, order]

def binupdate_c2(object[INT_t, ndim=1] bins, object[INT_t, ndim=1] cells,  object[INT_t, ndim=1] soilid, INT_t ncells, FLOAT_t dt, object[FLOAT_t, ndim=1] ksref, object[FLOAT_t, ndim=1] Dref, INT_t spltf, INT_t lbin):

    cdef INT_t ix
    cdef np.ndarray[INT_t, ndim=1] idx
    cdef np.ndarray[INT_t, ndim=1] order
    cdef np.ndarray[INT_t, ndim=1] order_n
    cdef np.ndarray[INT_t, ndim=1] bins_n

    order = np.arange(len(bins), dtype=np.intp)
    for ix in np.arange(ncells):
        idx = np.where(cells == ix)[0]
        [bins_n, order_n] = binupdate_c(bins[idx], dt, ksref[soilid[ix]], Dref[soilid[ix]], spltf, lbin)
        bins[idx] = bins_n
        order[idx] = order_n

    return [bins, order]