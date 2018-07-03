import numpy as np

def filmflow_t(particles_mzid, proj_mzid, ux, old_z, macdepth, s_samp, mfilling, lmx, mxgridcell, dt):
    from filmflow_c import firstfreeX

    s_red = np.zeros(lmx)
    #exfilt_retard = np.zeros(lmx)
    t_left = np.ones(lmx)*dt
    #contactfac = np.zeros(lmx)

    for i in s_samp:
        try:
            filmstep=firstfreeX(particles_mzid[i],proj_mzid[i],mfilling, mxgridcell)
        except:
            print('PROBLEM.')
            print(particles_mzid[i],proj_mzid[i])
            input("Press Enter to continue...")
        s_red[i]+=np.fmin(macdepth[filmstep]-old_z[i],0.)
        t_left[i]-=np.fmax(s_red[i]/ux[i],0.)
        filstep=np.fmin(filmstep,mxgridcell-particles_mzid[i]) #project step to end of film
        mfilling[particles_mzid[i]]-=1
        particles_mzid[i]=np.fmin(particles_mzid[i]+filstep,mxgridcell-2)
        mfilling[particles_mzid[i]]+=1
        proj_mzid[i]=np.fmin(proj_mzid[i]+filstep,mxgridcell-2)
    
    return [particles_mzid,proj_mzid,mfilling,s_red,t_left]