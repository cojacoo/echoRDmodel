import numpy as np
import scipy as sp
import scipy.stats as sps
import pandas as pd

def pmx_infilt(ti,precip,prec_part,acc_mxinf,thS,mc,pdyn,dt,prec_leftover=0,prec_2D=False,lastidx=0,method='MDA',infiltscale=False,npart=np.empty(0)):
    '''Infiltration Routine for echoRD Model
       (cc) jackisch@kit.edu 2016

       A) read rainfall input and convert into particles
       B) calculate infiltration domain states
       C) redistribution to macropore/matrix

       The routine can employ two methods at this stage by setting the method parameter: 
       MDA >> Based on Weiler 2003 the concept of macropore drainage areas is adapted and redistributes 
       incoming particles to near and open macropores. Infiltration is limited by macropore transport capacity.
       MED >> Based on the assumption of maximal export of free energy a second hypothesis redistributes 
       incoming particles hierarchically to the macropores based on their free capacity. Infiltration is
       limited by prec .le. k(infilt).

       INPUTS
       ti: time [s]
       dt: time step [s]
       precip: pandas data frame of precipitation as time series or reference table
               mc.prects specifies if this is a time series (True) or reference table (False)
       prec_part: accumulated precipitation which has not yet reached the mass of one full particle
       mc: parameters and references of echoRD model
       pdyn: particle dynamics routines of echoRD model
       pref_leftover: ponded water not yet infiltrated
       prec_2D: depending on the model utilisation precitiptation is converted to particles. 
                True if input is given in volume to whole domain [m3/s], False when given in [m/s]
       lastidx: last index of particle domain to give unique and traceable particle IDs
       method: method of redistribution of infiltrating particles (see above)

       OUTPUTS
       particles_infilt: pandas data frame of new particles to be concatenated to particle data frame
       prec_part: precipitation below the mass/volume of one particle for accumulation
       acc_mxinf: infiltration accumulation (important for very small time steps)
    '''
    
    # actual temperature of incoming water
    # DEBUG: handle later w/ time series
    T=np.array(9)

    # get timestep in prec time series and referenced precipitation
    if mc.prects==True:
        prec_id=np.argmin([np.abs(precip.index[x]-ti) for x in range(len(precip))])
        if precip.intense.iloc[prec_id]>0.:
            if prec_2D:
                #precip as m3/s
                prec_part+=precip.intense.values[prec_id]*dt/mc.particleV
            else:
                #precip as m/s
                prec_part+=precip.intense.values[prec_id]*dt*mc.mgrid.width.values/mc.particleA
            prec_avail=np.floor(prec_part)
            prec_part-=prec_avail
            prec_avail=int(prec_avail)
            
            prec_c=precip.conc.values[prec_id]
        else:
            prec_avail=0
            prec_c=0.
    #elif ((mc.prects=='column') | (mc.prects=='column2')):
    elif mc.prects=='column':
        prec_id=np.where((precip.tstart<=ti) & (precip.tend>ti))[0]
        if np.size(prec_id)>0:
            #prec_part+=precip.intense.values[prec_id]*dt/(36.*mc.particleV) #get true particle number for 5 degree open circle segment
            prec_part+=precip.intense.values[prec_id]*dt/mc.particleV
            prec_avail=np.floor(prec_part)
            prec_part-=prec_avail
            prec_avail=int(prec_avail)
            #avail. water particles
            prec_c=precip.conc.values[prec_id[0]]
        else:
            prec_avail=0
            prec_c=0.
    else:
        prec_id=np.where((precip.tstart<=ti) & (precip.tend>ti))[0]
        if np.size(prec_id)>0:
            #prec_avail=np.round(precip.intense.values[prec_id]*dt*mc.refarea*waterdensity(T,np.array(-9999))/mc.particlemass)
            if prec_2D:
                #precip as m3/s
                prec_part+=precip.intense.values[prec_id]*dt/mc.particleV
            else:
                #precip as m/s
                prec_part+=precip.intense.values[prec_id]*dt*mc.mgrid.width.values/mc.particleA
                #maybe a third option is needed if precip is given as kg/s
                #prec_part+=precip.intense.values[prec_id]*dt*1000./mc.particlemass #1000. to convert to g->kg
            prec_avail=np.floor(prec_part)
            prec_part-=prec_avail
            prec_avail=int(prec_avail)
            
            #avail. water particles
            prec_c=precip.conc.values[prec_id[0]]
        else:
            prec_avail=0
            prec_c=0.

    # reset particle definition in infilt container
    prec_potinf=prec_avail+prec_leftover
    
    if prec_potinf>0:
        particles_infilt=pd.DataFrame(np.zeros(prec_potinf*11).reshape(prec_potinf,11),columns=['lat', 'z', 'conc', 'temp', 'age', 'flag', 'fastlane', 'advect','cell','LTEbin','exfilt'])
        # place particles at surface and redistribute later according to ponding
        particles_infilt.z=-0.00001
        particles_infilt.lat=np.random.rand(prec_potinf)*mc.mgrid.width.values
        particles_infilt.conc=prec_c
        particles_infilt.temp=T
        particles_infilt.age=ti
        particles_infilt.flag=0
        particles_infilt.fastlane=np.random.randint(len(mc.t_cdf_fast.T), size=prec_potinf)
        particles_infilt.advect=0.

        #cases for single or multiple macropore configuration
        if mc.nomac=='Single':
            #single macropore defined
            idx_adv=prec_potinf #all particles selected
            particles_infilt.lat=mc.md_pos[0] #mc.mgrid.width[0]/2.
            particles_infilt.flag=1
            particles_infilt.advect=pdyn.assignadvect(prec_potinf,mc,particles_infilt.fastlane.values,True)
            activem=np.array([True])

        elif mc.nomac==True:
            idx_adv=prec_potinf #all particles selected
            particles_infilt.lat=mc.mgrid.width[0]/2.#mc.md_pos[0] #all into first cell
            particles_infilt.flag=1
            #particles_infilt.advect=pdyn.assignadvect(prec_potinf,mc,particles_infilt.fastlane.values,True)
            activem=np.array([False])

        elif type(mc.nomac)==float:
            #single macropore defined
            idx_adv=prec_potinf #all particles selected
            particles_infilt.lat=mc.md_pos[0] #all into first cell
            particles_infilt.flag=1
            particles_infilt.advect=pdyn.assignadvect(prec_potinf,mc,particles_infilt.fastlane.values,True)
            activem=np.array([True])

        elif mc.nomac!=True:
            # assign to different macropores
            particles_infilt.advect=pdyn.assignadvect(prec_potinf,mc,particles_infilt.fastlane.values,True)
            #a=True
            #activem=np.repeat(a,len(mc.md_pos)) #DEBUG: make dynamic!
            
            maxfill_top=np.ceil((mc.md_area/(-mc.gridcellA.values*mc.mgrid.latgrid.values))*mc.part_sizefac)[:,0]
            openslots=maxfill_top-mc.mactopfill
            activem=openslots>0
            idx_red=0

            if (((method=='MDA') | (method=='rand')) & any(activem)):
                # redistribution to macropores based on macropore drainage area
                # first layer as contact layer to surface
                # take cell numbers (binned data) and allow all but one for free drain
                if infiltscale:
                    nearby=(mc.mgrid.width/(2.*len(mc.md_pos)))*infiltscale
                    slots=[]
                    for k in np.where(activem)[0]:
                        #slots=np.concatenate([slots,np.arange(mc.md_pos[k]-nearby,mc.md_pos[k]+nearby,mc.particleD)])
                        slots=np.concatenate([slots,np.arange(mc.md_pos[k]-nearby,mc.md_pos[k]+nearby,mc.particleA/pdyn.filmthick(0))])
                    slots[slots>mc.mgrid.width.values]=slots[slots>mc.mgrid.width.values]-mc.mgrid.width.values
                    redist=np.random.randint(len(slots), size=prec_potinf)
                    particles_infilt.lat=slots[redist]
                
                #DEBUG: somehow other cells were assigned. this avoids that by making use of the cyclic boundary
                cellsx=pdyn.cellgrid(particles_infilt.lat,particles_infilt.z,mc).astype(int)
                cellsx[cellsx>=np.shape(npart)[1]]=np.fmin(np.shape(npart)[1],cellsx[cellsx>=np.shape(npart)[1]]-np.shape(npart)[1])
                particles_infilt.cell=cellsx
                
                ucellfreq=sps.itemfreq(particles_infilt.cell.values)
                freeparts=np.sum(ucellfreq[ucellfreq[:,1]>0.,1]-1.)
                if infiltscale:
                    freeparts+=np.sum(ucellfreq[ucellfreq[:,1]==1.,1])*(1.-infiltscale)
                
                # select number of free particles from particles_infilt at random (without proper reference of their position as it was there at random anyways)
                idx_adv=np.random.randint(prec_potinf,size=int(freeparts))
                if method=='MDA':
                    idx_red=macredist(particles_infilt.lat.values[idx_adv],mc,activem)
                else:
                    if any(activem==True):
                        # random infiltration into active macropores
                        idx_red=np.random.choice(np.where(activem==True)[0],int(freeparts))
                    else:
                        idx_red=0
            elif method=='MED':
                # infiltration based on maximum free energy dissipation
                [mx_infp,idx_red,acc_mxinf]=maxEinf(prec_potinf,acc_mxinf,thS,dt,mc)
                particles_infilt.flag.iloc[0:mx_infp]=0
                idx_adv=np.arange(prec_potinf)[mx_infp:prec_potinf]
            else:
                idx_adv=[]
            
            # assign incidences to particles
            particles_infilt.flag.iloc[idx_adv]=idx_red
            try:
                particles_infilt.lat.iloc[idx_adv]=mc.md_pos[idx_red-1]+(np.random.rand(len(idx_adv))-0.5)*mc.mgrid.vertfac.values
            except:
                print('no infilt reassignment. active macropores:')
                print(activem)

        #DEBUG: somehow other cells were assigned. this avoids that by making use of the cyclic boundary
        cellsx=pdyn.cellgrid(particles_infilt.lat,particles_infilt.z,mc).astype(int)
        cellsx[cellsx>=np.shape(npart)[1]]=np.fmin(np.shape(npart)[1],cellsx[cellsx>=np.shape(npart)[1]]-np.shape(npart)[1])
        particles_infilt.cell=cellsx

        # assign LTE bin - the new definition of the particle's state
        particles_infilt.LTEbin=getLTEbin(mc,npart,particles_infilt)

        # excess particles redistribute to macropores
        idx_ex=(particles_infilt.LTEbin>(len(mc.ku)-1))
        if ((len(idx_ex)>0) & any(activem)):
            particles_infilt.loc[idx_ex,'LTEbin']=len(mc.ku)-1
            particles_infilt.loc[idx_ex,'flag']=np.random.choice(mc.maccols[activem]+1,size=np.sum(idx_ex))

        #if any(particles_infilt.cell<0):
        #    print 'cell error at infilt'

        particles_infilt.index+=lastidx+1

    else:
        particles_infilt=pd.DataFrame([])

    #handle infiltration water as such, that it is prepared to take part in the standard operation
    #a particle is about 1 mm diameter at ks of 10-4m/s a time step of about 10 seconds is maybe a good start
    #maybe time step should be controlled through the actual maximal conductivity?
    return [particles_infilt,prec_part,acc_mxinf]


def macredist(lat,mc,activem):
    '''Distribute infiltration to macropores according to macropore drainage area
       Input: lateral position array, mc, bool mask of active macropores, mc.mgrid
       Output: index vector which macropore was appointed
    '''
    m_dist=np.diff(np.append(mc.md_pos[activem],mc.mgrid.width+mc.md_pos[0]))/2
    rightbound=mc.md_pos[activem]+m_dist
    activemacs=np.where(activem==True)
    latid=np.zeros(lat.shape, dtype=int)
    for i in np.arange(len(rightbound)):
        if i==0:
            macid=np.where((lat<rightbound[0]) | (lat>=rightbound[-1]))
        else:
            macid=np.where((lat<rightbound[i]) & (lat>=rightbound[i-1]))
    
        latid[macid]=activemacs[0][i]
    
    return latid


def maxEinf(prec_potinf,acc_mxinf,thS,dt,mc):
    '''Handle infiltration as maximum possible flux redistributed to macropores with 
       greatest capacity.
       Input:
       Output:
    '''
    #potential infiltration
    mac_infp=(mc.md_contact[:,0]/np.pi)*dt*-mc.a_velocity_real[-1]/mc.particleA
    mx_infp=(1.-mc.macshare[1])*mc.mgrid.width.values*np.mean(dt*mc.ku100[thS[0,:],mc.soilgrid[0,:]-1])/mc.particleA
    tot_infp=sum(mac_infp)+mx_infp
    acc_mxinf+=prec_potinf*(mx_infp/tot_infp)
    #if type(acc_mxinf)==pd.Series:
    #    acc_mxinf=acc_mxinf.values[0] #DEBUG: check why this happens
    mx_inf=int(np.round(acc_mxinf))
    acc_mxinf-=mx_inf
    
    mac_inf=np.zeros(len(mac_infp),dtype=int)
    prec_potinf-=mx_inf
    idx_red=np.zeros(prec_potinf,dtype=int)
    k=0
    i=0
    while (prec_potinf>0):
        mac_inf[i]=int(np.amin([np.round(mac_infp[i]/2.),prec_potinf]))
        prec_potinf-=mac_inf[i]
        idx_red[k:mac_inf[i]]=i+1
        k+=mac_inf[i]
        i+=1
        if i>=len(mac_infp):
            mx_inf+=prec_potinf
            prec_potinf=0

    return [mx_inf,idx_red,acc_mxinf]

def getLTEbin(mc,npart,particles_infilt):
    cells=particles_infilt.cell.values
    if ((np.shape(npart)[0]==0)&(mc.LTEdef!='ks')):
        print('Current state not given. Cannot calculate bins of new particles. Return NaN.')
        return np.nan
    if mc.LTEdef=='instant':
        #instant LTE is assumed. hence the particle receives the bin according to the new theta in the given cell
        if int(np.__version__.split('.')[0]+np.__version__.split('.')[1])<=18:
            infiltcells=np.unique(cells)
            counts=np.bincount(cells)
            counts=counts[counts>0]
        else:
            infiltcells, counts=np.unique(cells, return_counts=True)        
        npart[0,infiltcells]+=counts #update npart
        return npart[0,cells]
    elif mc.LTEdef=='ks':
        #initial infiltration into the coarse pores is assumed. hence the particel are set to the largest bins.
        return mc.mxbin[cells]-2 #simply return the maximum bin
    elif mc.LTEdef=='random':
        #new particles draw form empty bins at random.
        empty_brace=mc.mxbin[cells]-npart[cells]
        empty_brace*=np.random.random(len(empty_brace))
        return empty_brace.astype(int)+npart[0,cells]



# !!!
# calculate capacity of macropore at timestep > no; this will be solved dynamically
# check against redistribution capacity > means high intensity will clogg, low intensities will drain...
# ??redistribution capacity > slope, roughness
