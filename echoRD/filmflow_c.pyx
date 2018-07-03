import numpy as np
cimport cython

cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPEf = np.float
ctypedef np.float_t DTYPEf_t

#DTYPEb = np.bool
#ctypedef np.bool_t DTYPEb_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def firstfreeX(int idx, int idy, object[DTYPE_t, ndim=1] mfilling, int mxgridcell):
    #find first free slot on course and count steps
    cdef np.ndarray[DTYPE_t, ndim=1] f
    cdef int ff

    f=np.where(mfilling[idx:idy]==0)[0]
    if type(f)==int:
        if (idy-idx)>0:
            ff=np.argmin(mfilling[idx:idy])+idx
        else:
            ff=0
    elif(len(f)==0):
        ff=idx
    else:
        ff=np.amin([f[0],idy-idx])+idx

    ff = np.fmax(np.fmin(ff, mxgridcell-1),0)
    return ff

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def filmthick(float Psi):
    cdef float a,b,thick
    '''Calculate film thickness in fracture after Tokunaga & Wan 1997
       Psi: matric head in hPa
       Return: Thickness in metre
    
       Parameters derived by:
       psim=np.array([-12,-29,-42,-70,-92,-118,-167,-218,-316])
       filmthick=np.array([70,30,20,9,7,5,3,1.5,0.5])
       def func(x, a, b):
           return -a / x + b
       popt, pcov = curve_fit(func, psim[::-1], filmthick[::-1], maxfev = 1000000)

       Bound to Psi=[-3 .. -500]
    '''
    a=876.55078
    b=-2.18487753
    thick=-a / np.fmin(Psi,-3) + b
    return np.fmax(thick*1e-6,0.3e-6)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

#[particles_mzid,proj_mzid,mfilling,s_red,t_left]
#filmflow_c(particles_mzid, proj_mzid, ux, old_z, macdepth, s_samp, mfilling, len(samplenow), mxgridcell, dt)
def filmflow_c(object[DTYPE_t, ndim=1] particles_mzid, object[DTYPE_t, ndim=1] proj_mzid, object[DTYPEf_t, ndim=1] ux, object[DTYPEf_t, ndim=1] old_z, object[DTYPEf_t, ndim=1] macdepth, object[DTYPE_t, ndim=1] s_samp, object[DTYPE_t, ndim=1] mfilling, int lmx, int mxgridcell, float dt):
    cdef int i 
    cdef np.ndarray[DTYPEf_t, ndim=1] s_red = np.zeros(lmx, dtype=DTYPEf)
    #cdef np.ndarray[DTYPE_t, ndim=1] exfilt_retard = np.zeros(lmx, dtype=DTYPE)
    cdef np.ndarray[DTYPEf_t, ndim=1] t_left = np.ones(lmx, dtype=DTYPEf)*dt
    #cdef np.ndarray[DTYPEf_t, ndim=1] contactfac = np.zeros(lmx, dtype=DTYPEf)

    for i in s_samp:
        filmstep=firstfreeX(particles_mzid[i],proj_mzid[i],mfilling,mxgridcell)
        s_red[i]+=np.fmin(macdepth[filmstep]-old_z[i],0.)
        t_left[i]-=np.fmax(s_red[i]/ux[i],0.)
        filstep=np.fmin(filmstep,mxgridcell-particles_mzid[i]) #project step to end of film
        mfilling[particles_mzid[i]]-=1
        particles_mzid[i]=np.fmax(np.fmin(particles_mzid[i]+filstep,mxgridcell-1),0)
        mfilling[particles_mzid[i]]+=1
        proj_mzid[i]=np.fmax(np.fmin(proj_mzid[i]+filstep,mxgridcell-1),0)
    
    return [particles_mzid,proj_mzid,mfilling,s_red,t_left]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def macmatrix_c(object[DTYPEf_t, ndim=1] psi_mac, object[DTYPEf_t, ndim=1] dtheta_mac, object[DTYPEf_t, ndim=1] dpsi_dtheta_mac, object[DTYPEf_t, ndim=1] k_mac, object[DTYPEf_t, ndim=1] D_mac, object[DTYPE_t, ndim=1] FC_excess, object[DTYPE_t, ndim=2] stretch, int lmx, float particleD):
    cdef np.ndarray[DTYPEf_t, ndim=1] exp_psi = np.zeros(lmx, dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] dpsi_dtheta = np.zeros(lmx, dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] k = np.zeros(lmx, dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] D = np.zeros(lmx, dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] Qx = np.zeros(lmx, dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] thetax = np.zeros(lmx, dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] thick = np.zeros(lmx, dtype=DTYPEf)

    cdef np.ndarray[DTYPE_t, ndim=1] xweights
    cdef np.ndarray[DTYPEf_t, ndim=1] xtheta
    cdef np.ndarray[DTYPEf_t, ndim=1] wpsi
    cdef np.ndarray[DTYPEf_t, ndim=1] xdpsi
    cdef np.ndarray[DTYPEf_t, ndim=1] wdpsi
    cdef np.ndarray[DTYPEf_t, ndim=1] xk
    cdef np.ndarray[DTYPEf_t, ndim=1] wk
    cdef np.ndarray[DTYPEf_t, ndim=1] wt
    cdef np.ndarray[DTYPEf_t, ndim=1] xD
    cdef np.ndarray[DTYPEf_t, ndim=1] wD

    #FC_exp=np.zeros(len(idx),dtype=bool)
    
    for i in np.arange(lmx):
        #reversely weighted geometric mean or mean if more appropriate
        xpsi=psi_mac[stretch[i][0]:(stretch[i][0]+stretch[i][1])]
        xtheta=dtheta_mac[stretch[i][0]:(stretch[i][0]+stretch[i][1])]
        xweights=np.arange(1+len(xpsi))[:0:-1]
        wpsi=np.repeat(xpsi,xweights)
        exp_psi[i]=-(np.abs(np.prod(wpsi))**(1./np.fmax(1,len(wpsi))))
        xdpsi=dpsi_dtheta_mac[stretch[i][0]:(stretch[i][0]+stretch[i][1])]
        wdpsi=np.repeat(xdpsi,xweights)
        thick[i]=filmthick(np.mean(wpsi)*98.0665) #convert m water column into hPa and calculate film thickness
        dpsi_dtheta[i]=np.prod(wdpsi)**(1./np.fmax(1,len(wdpsi)))
        xk=k_mac[stretch[i][0]:(stretch[i][0]+stretch[i][1])]
        wk=np.repeat(xk,xweights)
        wt=np.repeat(xtheta,xweights)
        k[i]=np.mean(wk)
        xD=D_mac[stretch[i][0]:(stretch[i][0]+stretch[i][1])]
        wD=np.repeat(xD,xweights)
        D[i]=np.mean(wD)
        thetax[i]=np.mean(wt)
        Qx[i]=np.mean(wD)*(np.mean(wt)/(0.5*particleD))+np.mean(wk)
        
        if any(FC_excess[stretch[i][0]:(stretch[i][0]+stretch[i][1])]==1): #flag fc_check is True and FC is met or excceeded > reduce probability to infiltrate by number of free slots
            #wfcx=FC_free_prob[stretch[i][0]:(stretch[i][0]+stretch[i][1])]
            #wfc=np.repeat(wfcx,xweights)
            #FC_exp[i]=np.random.random(1)<=np.mean(wfc)
            Qx[i]=0.

    return [Qx, dpsi_dtheta, exp_psi, thick, k, D, thetax]