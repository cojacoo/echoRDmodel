#echoRD_1D - a simple 1D version of the echoRD particle model
#(cc) jackisch@kit.edu
import numpy as np
import pandas as pd
import scipy as sp

def gridupdate_thS1D(cellid,mc,pdyn):
    '''Calculates thetaS from particle density
    '''
    cells=np.append(range(mc.mgrid.cells),cellid)
    cells[cells<0]=0
    cells[cells>len(mc.zgrid[:,1])]=len(mc.zgrid[:,1])
    thetaS=np.floor(100.*np.bincount(cellid).astype(float)/mc.part_sizefac)
    thetaS[thetaS>100.]=100.
    return thetaS.astype(np.int)

def part_diffusion_1D(particles,thS,mc,vG,pdyn,dt,uffink_corr=True):
    '''Calculate Diffusive Particle Movement
       Based on state in grid use diffusivity as foundation of 2D random walk.
       Project step and check for boundary conditions and further restrictions.
       Update particle positions.
    '''
    N=len(particles.z) #number of particles

    # 1D Random Walk function with additional correction term for
    # non-static diffusion after Uffink 1990 p.15 & p.24ff and Kitanidis 1994
    xi=np.random.rand(N)*2.-1.

    u=mc.ku[thS,mc.soilgrid[:,4]-1]/mc.theta[thS,mc.soilgrid[:,4]-1]
    D=mc.D[thS,mc.soilgrid[:,4]-1]*mc.soilmatrix.ts[mc.soilgrid[:,4]-1].values
 
    vert_sproj=(dt*u[particles.cell.values.astype(np.int)] + (xi*((2*D[particles.cell.values.astype(np.int)]*dt)**0.5)))

    if (uffink_corr==True):
        #Ito Scheme after Uffink 1990 and Kitanidis 1994 for vertical step
        #modified Stratonovich Scheme after Kitanidis 1994 for lateral step
        dx=vert_sproj
        
        # project step and updated state
        # new positions
        z_proj=particles.z.values-vert_sproj
        [lat_proj,z_proj,nodrain]=pdyn.boundcheck(particles.lat,z_proj,mc)
        thSx=gridupdate_thS1D(pdyn.cellgrid(particles.lat,z_proj,mc).astype(np.int64),mc,pdyn) 
        cell_proj=pdyn.cellgrid(particles.lat,z_proj,mc).astype(np.int64)
        
        u_proj=mc.ku[thSx,mc.soilgrid[:,4]-1]/mc.theta[thSx,mc.soilgrid[:,4]-1]
        D_proj=mc.D[thSx,mc.soilgrid[:,4]-1]*mc.soilmatrix.ts[mc.soilgrid[:,4]-1].values
    
        corrD=np.abs(D_proj[particles.cell.values.astype(np.int)]-D[particles.cell.values.astype(np.int)])/dx
        corrD[dx==0.]=0.
        D_mean=np.sqrt(D_proj[particles.cell.values.astype(np.int)]*D[particles.cell.values.astype(np.int)])
        corru=np.sqrt(u[particles.cell.values.astype(np.int)]*u_proj[particles.cell.values.astype(np.int)])

        vert_sproj=((corru-corrD)*dt + (xi*((2*D[particles.cell.values.astype(np.int)]*dt)**0.5)))


    # new positions
    z_new=particles.z.values-vert_sproj

    [particles.lat,particles.z,nodrain]=pdyn.boundcheck(particles.lat,z_proj,mc)
    particles['cell']=pdyn.cellgrid(particles.lat,particles.z,mc).astype(np.int64)

    # saturation check
    thS=gridupdate_thS1D(pdyn.cellgrid(particles.lat,z_new,mc).astype(np.int64),mc,pdyn) #DEBUG: externalise smooth parameter

    if any(-nodrain):
        particles.loc[-nodrain,'flag']=len(mc.maccols)+1

    return [particles,thS]

def part_advect_1D(particles,dt,pdyn,mc):
    idx=particles['flag']!=0
    if any(idx):
        particles.z.loc[idx]=particles.z.loc[idx]+particles.advect.loc[idx]*dt
        particles.cell.loc[idx]=pdyn.cellgrid(particles.lat.loc[idx],particles.z.loc[idx],mc).astype(np.int64).values
    return particles

def infilt_1Dobs(ti,precip,prec_part,mc,pdyn,dt):
    T=np.array(9)
    # get timestep in prec time series
    if mc.prects:
        prec_id=np.argmin([np.abs(precip.index[x]-ti) for x in range(len(precip))])
        if precip.iloc[prec_id]>0:
            prec_avail=int(np.round(precip.iloc[prec_id]/600.*dt*mc.part_sizefac/(-mc.mgrid.vertfac.values*1000.)))
            #avail. water particles
            prec_c=np.array([23.])
        else:
            prec_avail=0
            prec_c=0.
    else:
        prec_id=np.where((precip.tstart<=ti) & (precip.tend>ti))[0]
        if np.size(prec_id)>0:
            #prec_avail=np.round(precip.intense.values[prec_id]*dt*mc.refarea*waterdensity(T,np.array(-9999))/mc.particlemass)
            #prec_avail=int(np.round(precip.intense.values[prec_id]*dt*mc.part_sizefac/(-mc.mgrid.vertfac.values*1000.)))
            prec_part+=precip.intense.values[prec_id]*dt*mc.part_sizefac/(-mc.mgrid.vertfac.values*0.13)
            #prec_part+=precip.intense.values[0]*2.*mc.part_sizefac/(-mc.mgrid.vertfac.values*0.13) #only for debug
            prec_avail=np.floor(prec_part)
            prec_part-=prec_avail
            prec_avail=int(prec_avail)
            #avail. water particles
            prec_c=precip.conc.values[prec_id[0]]
        else:
            prec_avail=0
            prec_c=0.

    if prec_avail>0:
        particles_infilt=pd.DataFrame(np.zeros(prec_avail*10).reshape(prec_avail,10),columns=['lat', 'z', 'conc', 'temp', 'age', 'flag', 'fastlane', 'advect','lastZ','cell'])
        # place particles at surface and redistribute later according to ponding
        particles_infilt.z=-0.00001
        particles_infilt.lat=np.random.rand(prec_avail)*mc.mgrid.latfac.values
        particles_infilt.conc=prec_c
        particles_infilt.temp=T
        particles_infilt.age=ti
        particles_infilt.flag=1
        particles_infilt.fastlane=np.random.randint(len(mc.t_cdf_fast.T), size=prec_avail)
        particles_infilt.cell=0
        particles_infilt.advect=pdyn.assignadvect(prec_avail,mc,particles_infilt.fastlane.values,False)
    else:
        particles_infilt=pd.DataFrame([])

    return [particles_infilt,prec_part]

def CAOSpy_run1D_adv(particles,thS,leftover,drained,tstart,tstop,precTS,mc,pdyn,cinf,vG):
    timenow=tstart
    precparts=0
    prec_part=0.
    #loop through time
    while timenow < tstop:
        #define dt as Courant/Neumann criterion
        dt_D=(mc.mgrid.vertfac.values[0])**2 / (np.amax(mc.D[np.amax(thS),:]))
        dt_ku=-mc.mgrid.vertfac.values[0]/np.amax(mc.ku[np.amax(thS),:])
        dt=np.amin([dt_D,dt_ku])
        
        #INFILT
        [p_inf,prec_part]=infilt_1Dobs(timenow,precTS,prec_part,mc,pdyn,dt)
        particles=pd.concat([particles,p_inf])
        #psi=vG.psi_thst(thS/100.,mc.soilmatrix.alpha[mc.soilgrid[:,1]-1],mc.soilmatrix.n[mc.soilgrid[:,1]-1])
        #DIFFUSION
        [particles,thS]=part_diffusion_1D(particles,thS,mc,vG,pdyn,dt,uffink_corr=True)
        
        #ADVECTION
        particles=part_advect_1D(particles,dt,pdyn,mc)
        
        #drained particles
        drained=drained.append(particles[particles.flag==len(mc.maccols)+1])
        particles=particles[particles.flag!=len(mc.maccols)+1]
        pondparts=(particles.z<0.)
        leftover=np.count_nonzero(-pondparts)
        [particles.lat,particles.z,nodrain]=pdyn.boundcheck(particles.lat,particles.z,mc)
        particles['cell']=pdyn.cellgrid(particles.lat,particles.z,mc).astype(np.int64)
        thS=gridupdate_thS1D(particles.cell,mc,pdyn)
        timenow=timenow+dt
        precparts+=len(p_inf)

    return(particles,thS,leftover,drained,timenow,precparts)
