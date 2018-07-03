# coding=utf-8

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.ndimage as spn
import dataread as dr
import vG_conv as vG

#particle dynamics
#macropore mc.soilmatrix interaction
def cellgrid(lat,z,mc):
    '''Calculate cell number from given position of a particle.
    '''
    rw=np.floor(z/mc.mgrid.vertfac.values)
    cl=np.floor(lat/mc.mgrid.latfac.values)
    cell=rw*mc.mgrid.latgrid.values + cl
    #if len(np.unique(cell))!=mc.mgrid.cells.values:
    #    print 'runaway particles'
    #if any(np.isnan(cell)):
    #    A=np.where(np.isnan(cell))
    #    print 'R/C',rw[A],cl[A]
    #    print 'z/lat',z[A],lat[A]

    cell[cell<0.]=0.
    cell[cell>=mc.mgrid.cells.values[0]]=mc.mgrid.cells.values[0]-1.

    return cell.astype(np.int64)


def gridupdate(particles,mc):
    '''Calculates grid state from particle density. DEPRECATED
    '''
    npart=np.zeros((mc.mgrid.vertgrid,mc.mgrid.latgrid), dtype=int64)*(2*mc.part_sizefac)
    npart[np.unravel_index(np.arange(mc.mgrid.cells.astype(int64)),(mc.mgrid.vertgrid.astype(int64),mc.mgrid.latgrid.astype(int64)))] = np.bincount(particles.cell)
    #theta=npart*mc.particleA/mc.mgrid.gridcellA
    return npart


def thetaSid(npart,mc):
    '''Calculates thetaS id from npart
       Use the result as: mc.D[thetaSid(npart,mc.soilmatrix,mc,mc.soilgrid).astype(int),mc.soilgrid.ravel()-1]
       on prepared D, ku, theta, psi which are stored in mc
    '''
    thetaS=npart.ravel()/(mc.soilmatrix.ts[mc.soilgrid.ravel()-1]*(2*mc.part_sizefac))
    return thetaS*100

def thetaSidx(npart,mc):
    '''Calculates thetaS id from npart and cuts at max.
       WARNING: oversaturation and draining will simply be accepted!!!
       Use the result as: mc.D[thetaSid(npart,mc.soilmatrix,mc,mc.soilgrid).astype(int),mc.soilgrid.ravel()-1]
       on prepared D, ku, theta, psi which are stored in mc
    '''
    thetaS=npart.ravel()/(mc.soilmatrix.ts[mc.soilgrid.ravel()-1]*(2*mc.part_sizefac))
    thetaS[thetaS>1.]=1.
    thetaS[thetaS<0.]=0.
    return thetaS*100


def gridupdate_thS(lat,z,mc,gauss=0.5):
    '''Calculates thetaS from particle density
    '''
    #import numpy as np
    #import scipy as sp
    #import scipy.ndimage as spn
    npart=np.zeros((mc.mgrid.vertgrid,mc.mgrid.latgrid), dtype=np.int64)*(2*mc.part_sizefac)
    lat1=np.append(lat,mc.onepartpercell[0])
    z1=np.append(z,mc.onepartpercell[1])
    cells=cellgrid(lat1,z1,mc)
    #DEBUG:
    cells[cells<0]=0
    trycount = np.bincount(cells)
    trycount=trycount-1 #reduce again by added particles
    npart[np.unravel_index(np.arange(mc.mgrid.cells.astype(np.int64)),(mc.mgrid.vertgrid.values.astype(np.int64),mc.mgrid.latgrid.values.astype(np.int64)))] = trycount
    #npart_s=spn.filters.median_filter(npart,size=mc.smooth)
    #npart_s=spn.filters.gaussian_filter(npart,gauss)
    #do not smooth at macropores centroids
    #npart_s[np.unravel_index(mc.maccells,(mc.mgrid.vertgrid.values.astype(np.int64),mc.mgrid.latgrid.values.astype(np.int64)))]=npart[np.unravel_index(mc.maccells,(mc.mgrid.vertgrid.values.astype(np.int64),mc.mgrid.latgrid.values.astype(np.int64)))]
    #thetaS=npart_s.ravel()/(mc.soilmatrix.ts[mc.soilgrid.ravel()-1]*(2*mc.part_sizefac))
    if mc.prects=='column':
        if mc.colref==False:
            #initialise reference
            #Amacropore=np.pi*0.005**2
            #Arings=np.pi*((np.arange(mc.mgrid.latgrid.values/2)+1.)*mc.mgrid.latfac.values)**2
            #tsrefline=np.diff(np.append(0.,Arings))/4. #quater circle areas
            #mc.tsref=np.tile(np.append(tsrefline[::-1],tsrefline),mc.mgrid.vertgrid).reshape(np.shape(mc.soilgrid))
            #mc.trref=mc.tsref
            #mc.tsref*=mc.soilmatrix.ts[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid))*abs(mc.mgrid.vertfac.values)
            #mc.trref*=mc.soilmatrix.tr[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid))*abs(mc.mgrid.vertfac.values)

            circumsegment=(np.arange(mc.mgrid.latgrid.values/2)+1.)*mc.mgrid.latfac.values*np.pi/(360./5.)
            mc.tsref=np.tile(np.append((mc.particleD/circumsegment)[::-1],(mc.particleD/circumsegment)),mc.mgrid.vertgrid).reshape(np.shape(mc.soilgrid))
            #this is the VOLUME of saturation in the respective cell of the half cylinder
        ths_part=mc.soilmatrix.ts[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid))*(2*mc.part_sizefac)
        #thetaS=npart.astype(np.float)*mc.particleV/mc.tsref
        thetaS=mc.tsref*npart.astype(np.float)/ths_part
    elif mc.prects=='column2':
        if mc.colref==False:
            macref=mc.particleD/(0.005*np.pi)
            circumsegment=mc.particleD/((np.arange(mc.mgrid.latgrid.values/2)+1.)*mc.mgrid.latfac.values*np.pi)
            mc.moistfac=np.tile(np.append(circumsegment[::-1],circumsegment),mc.mgrid.vertgrid).reshape(np.shape(mc.soilgrid))
            #this is the VOLUME of saturation in the respective cell of the half cylinder
        ths_part=mc.soilmatrix.ts[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid))*(2*mc.part_sizefac)
        #thetaS=npart.astype(np.float)*mc.particleV/mc.tsref
        thetaS=mc.moistfac*npart.astype(np.float)/ths_part
    else:
        ths_part=mc.soilmatrix.ts[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid))*(2*mc.part_sizefac)
        thetaS=npart.astype(np.float)/ths_part
    thetaS[thetaS>0.99]=0.99
    thetaS[thetaS<0.1]=0.1
    return [(thetaS*100).astype(np.int),npart]


def npart_theta(npart,mc):
    '''Calculates theta from npart
    '''
    #import numpy as np
    #import scipy as sp
    #import scipy.ndimage as spn
    npart_s=spn.filters.median_filter(npart,size=mc.smooth)
    theta=(mc.particleA/(-mc.gridcellA))[0]*npart_s.ravel()
    return theta

def boundcheck(lat,z,mc):
    '''Boundary checks
    '''
    #cycle bound:
    #if any(lat<0.0):
    lat[lat<0.0]=mc.mgrid.width[0]+lat[lat<0.0]
    #if any(lat>mc.mgrid.width[0]):
    reffac=np.floor(lat[lat>mc.mgrid.width[0]]/mc.mgrid.width[0])
    lat[lat>mc.mgrid.width[0]]=lat[lat>mc.mgrid.width[0]]-mc.mgrid.width[0]*reffac
    #if any(lat<0.0):
    reffac=np.floor(np.abs(lat[lat<0.0]/mc.mgrid.width[0]))
    lat[lat<0.0]=lat[lat<0.0]+mc.mgrid.width[0]*reffac
    #topbound - DEBUG: set zero for now, may interfere with infilt leftover idea
    #if any(z>-0.00001):
    z[z>-0.00001]=-0.00001

    #SAFE MODE
    #if any(np.isnan(z)):
    #    print 'z went NaN > pushed back into domain'
    #    z[np.isnan(z)]=-0.5
    #    exit()
    #if any(np.isnan(lat)):
    #    print 'lat went NaN > pushed back into domain'
    #    lat[np.isnan(lat)]=0.1
    #    exit()
    
    #lowbound - leave to drain DEBUG: may be a case parameter!!!
    #DEBUG: maybe allow ku defined flux
    nodrain=(z>=mc.mgrid.depth[0])
    z[-nodrain]=mc.mgrid.depth[0]+0.000000000001
    return [lat,z,nodrain]

def assignadvect(no,mc,dummy=None,realcrosssec=True):
    '''Assign advective velocity from observed velocity distribution 
       stored in mc.a_velocity or mc.a_velocity_real.
       Alternatively assign a measured literature value.
    '''
    #advective velocity in m/s
    if mc.advectref=='Shipitalo':
        adv=np.array([-0.0676]).repeat(no) #Shipitalo and Butt (1999)
    elif mc.advectref=='Weiler':
        adv=np.array([-0.0774]).repeat(no) #Weiler (2001)
    elif mc.advectref=='Zehe':
        adv=np.array([-0.058]).repeat(no) #Zehe and Flühler (2001)
    elif mc.advectref=='geogene':
        adv=np.array([-0.774]).repeat(no) #scaled 10x Weiler (2001)
    elif mc.advectref=='geogene2':
        adv=np.array([-7.74]).repeat(no) #scaled 100x Weiler (2001)
    elif mc.advectref=='obs':
        if dummy==None: #fast lane not given
            dummy=np.random.randint(len(mc.t_cdf_fast.T), size=no)

        dummx=np.random.rand(no)
        cum_cdf=np.array(mc.t_cdf_fast.cumsum(axis=0))
        l_cdf=cum_cdf.shape[0]
        idx=abs(np.floor(cum_cdf[:,dummy]-dummx.repeat(l_cdf).reshape(cum_cdf[:,dummy].T.shape).T).sum(axis=0)).astype(np.int)
        if realcrosssec:
            adv=mc.a_velocity_real[idx]
        else:
            adv=mc.a_velocity[idx]

        #velocities
        #this was earlier not really well cited!
        #still not quite clear if this refers to u_max or u_mean
        #Zehe and Flühler: D 2-4mm  4-6mm  6-8mm  8-10mm
        #                  v 6.5e-3 1.8e-2 3.5e-2 5.8e-2 [m/s]
        #                  Q 4.6-8  3.5e-7 1.4e-6 3.8e-6 [m3/s]

        #Weiler: [6.2, 2.9, 3.5, 5.7] (all cm3/s) > e-6 [m3/s] #experiments backing the values: [4,2,4,6]
        #Weiler weighted mean: (np.sum(np.array([6.2e-6, 2.9e-6, 3.5e-6, 5.7e-6])*np.array([4.,2.,4.,6.]))/16.)/(np.pi*0.0045**2.)=0.077416108121242916
        #Wang et al 94 and Bouma et al 82: 6.5 > 0.1 m/s
        #shipitalo Gibbs 2000: 2.6
        #Shipitalo and Butt 99: 4.3 this is also not quite well cited

    return adv

def mac_advection(particles,mc,thS,dt,clog_switch=False,maccoatscaling=1.,exfilt_method='Ediss',film=True,retardfac=0.5,dynamic_pedo=False,ksnoise=1.):
    '''Calculate Advection in Macropore
       Advective particle movement in macropore with retardation of the advective momentum 
       through drag at interface, plus check for each macropore's capacity and possible overload (clog_switch).

       INPUTS
       particles: pandas data frame of all particles
       mc: echoRD parameters
       thS: 2D array with state of matrix grid
       dt: time step [s]
       clog_switch: if True, an overshoot of the macropore capacity will lead to clogging
       maccoatscaling: scaling factor for the experienced drag from the matrix handled
       exfilt_method: method of exfiltration calculation from the macropore
                      Ediss = Energy Dissipation: particle retardation is based on additional interaction with the matrix handled
                      RWdiff = Random Walk Diffusion: a simple stochastic term simulates diffusion into the matrix
       film: if True, particles may move more quickly if inside a film at the pore wall
       retardfac: assumption of wetting resistance - factor reducing infiltration of the first film layer (only if film == True)

       OUTPUTS
       particles: pandas data frame of all particles after advection
       s_red_store: array with all advective steps taken
       exfilt_p: number of particles which exfiltrated from the macropores
    '''

    pm=mc.particlemass/1000. #particle mass conversion into kg

    thS=thS.ravel()
    s_red=np.array([])
    exfilt_p=0.
    s_red_store=[]
    refpos=np.unique(mc.macP[0].exterior.coords.xy[1])[::-1] #reference position (z) of macropore capacity > this may be sourced in dataread
    refpos=np.round(-refpos/mc.particleD).astype(int) #reference position as macropore index
    #loop through macropores
    for maccol in np.arange(len(mc.maccols)):
        macstate=[]
        if not particles.loc[particles.flag==(maccol+1)].empty:
            #project advection step
            midx=np.where(particles.flag==(maccol+1))[0]
            #macropore is divided in grid of particle diameter steps
            #ux=particles.loc[particles.flag==(maccol+1),'advect'].values
            
            #id and filling in macropore grid
            mxgridcell=np.floor(mc.md_macdepth[maccol]/mc.particleD).astype(np.int64)[0]
            def macpos(z,mxgridcell):
                #get position in macropore
                p_mzid=np.floor(-z/mc.particleD).astype(int)
                #bound checks
                p_mzid[p_mzid>mxgridcell-1]=mxgridcell-1
                p_mzid[p_mzid<0]=0
                #Outputs: 1 film id
                return p_mzid

            def macfil(p_mzid,mxgridcell):
                #get macropore filling and position in film
                p_mzid_plus1=np.append(p_mzid,np.arange(mxgridcell))
                mfilling=np.bincount(p_mzid_plus1)-1
                filmloc=np.ones(len(p_mzid),dtype=int)
                for idx in np.where(mfilling>0)[0]:
                    idy=np.where(p_mzid==idx)
                    filmloc[idy]=np.arange(mfilling[idx],dtype=int)+1
                #Outputs: 1 filling state, 2 location in film/distance to porewall
                return [mfilling, filmloc]

            #find macropore increment of particles
            def findincr(x):
                #DEBUG
                #idxx=0
                idx=np.where(mc.md_depth>-x)[0]-1
                if len(idx)>=1:
                    idxx=idx[0]
                else:
                    idxx=len(mc.md_depth)-1
                return idxx
            vfindincr = np.vectorize(findincr)

            #id in macropore
            particles_mzid=macpos(particles.loc[particles.flag==(maccol+1),'z'].values,mxgridcell)
            
            [mfilling, filmloc]=macfil(particles_mzid,mxgridcell)
            
            #advective velocity
            ux=particles.loc[particles.flag==(maccol+1),'advect'].values
            #particles reset advective velocity when far from pore wall
            ux[filmloc>1]=assignadvect(sum(filmloc>1),mc)

            s_proj=ux*dt #project step
            z_proj=particles.loc[particles.flag==(maccol+1),'z'].values+s_proj #project new position

            #check lower boundary
            nodrain=(z_proj>=mc.soildepth)
            if any(-nodrain):
                z_proj[-nodrain]=mc.soildepth
            #cell of projected step
            proj_mzid=macpos(z_proj,mxgridcell)

            #id of macropore in soil grid
            mac_cell=cellgrid(mc.md_pos[maccol].repeat(mxgridcell),-np.arange(mxgridcell)*mc.particleD,mc)

            exfilt=particles_mzid<0 #create exfilt array with all False
            s_red_store=np.zeros(len(particles_mzid))
            
            #splitsample macropore particles according to filling
            splitfac=int(np.round(np.amax(mfilling).astype(float)/2.))
            if splitfac>1:
                N_tot=np.arange(len(particles_mzid),dtype=np.int64)
                if (len(N_tot)/splitfac)*splitfac!=len(N_tot):
                    adddummy=int(np.round((np.ceil(len(N_tot)/float(splitfac))-len(N_tot)/float(splitfac))*splitfac))
                    N_tot=np.concatenate([N_tot,np.zeros(adddummy)*np.nan])
                sampleset=np.random.permutation(N_tot).reshape([splitfac,len(N_tot)/splitfac]).astype(np.int64)
            else:
                splitfac=1
                #sampleset=np.arange(len(particles_mzid),dtype=np.int64)
                sampleset=np.tile(np.arange(len(particles_mzid),dtype=np.int64),2).reshape([2,len(particles_mzid)])
            
            particles_znew=particles.loc[particles.flag==(maccol+1),'z'].values

            #loop through splitsamples:
            for subsample in np.arange(splitfac):
                samplenow=sampleset[subsample]
                #if type(sampleset[subsample])!=np.array:
                #    samplenow=np.array(sampleset[subsample])
                #elif len(sampleset[subsample])>1:
                samplenow=samplenow[samplenow>=0]
                if sum(samplenow>=0)<1:
                    continue #go to next iteration cycle if no particles are selected

                #update gridfill
                particles_mzid=macpos(particles_znew,mxgridcell)
                [mfilling, filmloc]=macfil(particles_mzid,mxgridcell)

                #functions to check free slots in macropore along path
                #mfilling must be defined before as filling state of macropore, thus new def of functions here
                def contactcount(idx,idy):
                    #ocuppied slots on course
                    return np.count_nonzero(mfilling[idx:idy]==0)
                vcontactcount=np.vectorize(contactcount)
                def freecount(idx,idy):
                    #free slots on course
                    return sum(mfilling[idx:idy]==0)
                vfreecount=np.vectorize(freecount)
                def firstfree(idx,idy):
                    #find first free slot on course and count steps
                    f=np.where(mfilling[idx:idy]==0)[0]
                    if len(f)==0:
                        ff=0
                    else:
                        f=np.amax(f[0]-1,0)
                        ff=np.fmin(f,idy-idx)
                    return ff
                vfirstfree=np.vectorize(firstfree)

                #dragweight=vcontactcount(particles_mzid[samplenow],proj_mzid[samplenow]) #free slots on course
                #filmweight=vfreecount(particles_mzid[samplenow],proj_mzid[samplenow]) #occupied slots on course
                #passage=(proj_mzid[samplenow]-particles_mzid[samplenow]).astype(np.float64) #length of projected voyage

                #soil cell id of start and end
                idx=mac_cell[particles_mzid[samplenow]]
                idy=mac_cell[proj_mzid[samplenow]]
                s_red=np.zeros(len(samplenow))
                t_left=np.ones(len(samplenow))*dt
                #u_hag=s_red #advective velocity after hagen-poiseuille
                contactfac=np.ones(len(samplenow),dtype=np.float64)

                #project diffusion into matrix
                ##CONTACT FACE##
                if film:
                    #assume film initialisation at pore wall 
                    exfilt_retard=np.zeros(len(samplenow),dtype=int)
                    
                    #particles will proceed with v_adv to the end of the film
                    #therefore the reference will shift to the first free slot
                    ib=filmloc[samplenow]>1
                    if any(ib):
                        filmstep=vfirstfree(particles_mzid[samplenow[ib]],proj_mzid[samplenow[ib]])
                        s_red[ib]=-(filmstep+0.45)*mc.particleD
                        t_left[ib]=np.fmax(s_red[ib]/ux[samplenow[ib]],0.)

                        particles_mzid[samplenow[ib]]+=np.fmin(filmstep,mxgridcell-particles_mzid[samplenow[ib]]) #project step to end of film
                        idx=mac_cell[particles_mzid[samplenow]] #update reference to soil
                        
                    filmweight=vcontactcount(particles_mzid[samplenow],proj_mzid[samplenow]).astype(np.float64) #free slots on course
                    passage=(proj_mzid[samplenow]-particles_mzid[samplenow]).astype(np.float64) #length of projected voyage
                    
                    contactfac=np.ones(len(samplenow),dtype=np.float64)
                    ia=filmweight>0.
                    ic=passage>0.
                    if any(ia & ic):
                        contactfac[ia & ic]=filmweight[ia & ic]/passage[ia & ic]

                    #particles at position 1 in film can be retarded for exfiltration to simulate a film
                    exfilt_retard[filmloc[samplenow]==1]=1
                    
                else:
                    #assume only film particles to interact with the matrix
                    dragweight=vcontactcount(particles_mzid[samplenow],proj_mzid[samplenow]) #free slots on course
                    passage=(proj_mzid[samplenow]-particles_mzid[samplenow]).astype(np.float64) #length of projected voyage
                    ia=dragweight>0.
                    ib=passage>0.
                    if any(ia & ib):
                        contactfac[ia & ib]=1.-(dragweight[ia & ib]/passage[ia & ib])
                
                #scale contactfac with coating factor
                contactfac=contactfac/maccoatscaling

                if exfilt_method=='RWdiff':
                    xi=np.random.rand(len(samplenow))
                    #diffusion over projected passage as geo mean of start and end
                    if dynamic_pedo:
                        psi1=vG.psi_thst(thS.ravel()[idx],mc.soilmatrix.alpha[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.n[mc.soilgrid.ravel()[idx]-1]).values
                        psi2=vG.psi_thst(thS.ravel()[idy],mc.soilmatrix.alpha[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.n[mc.soilgrid.ravel()[idy]-1]).values
                        if type(ksnoise)==float:
                            D1=vG.D_psi(psi1,ksnoise*mc.soilmatrix.ks[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.ts[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.tr[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.n[mc.soilgrid.ravel()[idx]-1])
                            D2=vG.D_psi(psi2,ksnoise*mc.soilmatrix.ks[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.ts[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.tr[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.n[mc.soilgrid.ravel()[idy]-1])
                        else:
                            D1=vG.D_psi(psi1,ksnoise[idx]*mc.soilmatrix.ks[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.ts[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.tr[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.n[mc.soilgrid.ravel()[idx]-1])
                            D2=vG.D_psi(psi2,ksnoise[idy]*mc.soilmatrix.ks[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.ts[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.tr[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.n[mc.soilgrid.ravel()[idy]-1])
                        D=np.sqrt(D1*D2)
                    else:
                        D=np.sqrt(mc.D[thS.ravel()[idx],mc.soilgrid.ravel()[idx]-1]*mc.D[thS.ravel()[idy],mc.soilgrid.ravel()[idy]-1])

                    diff_proj=(xi*((2*D*t_left)**0.5))*contactfac

                    if film:
                        diff_proj[exfilt_retard==1]*=retardfac
                    adv_retard=(mc.particleD.repeat(len(diff_proj))-diff_proj)/mc.particleD
                    adv_retard[adv_retard<0.]=0.
                    
                    ux[samplenow]*=adv_retard
                    s_red+=ux[samplenow]*t_left
                    
                    exfilt[samplenow]=(adv_retard<=0.3)
                #elif exfilt_method=='Ediss':
                else:
                    #experienced psi
                    if dynamic_pedo:
                        xsample=mc.soilgrid.ravel()[idx]-1
                        psi1=vG.psi_thst(thS.ravel()[idx]/100.,mc.soilmatrix.alpha[xsample].values,mc.soilmatrix.n[xsample].values)
                        xsample=mc.soilgrid.ravel()[idy]-1
                        psi2=vG.psi_thst(thS.ravel()[idy]/100.,mc.soilmatrix.alpha[xsample].values,mc.soilmatrix.n[xsample].values)
                        exp_psi=-np.sqrt(psi1*psi2)
                        xsample=mc.soilgrid.ravel()[idx]-1
                        ysample=mc.soilgrid.ravel()[idy]-1
                        if type(ksnoise)==float:
                            dD1=vG.dcst_thst(thS.ravel()[idx]/100., mc.soilmatrix.ts[xsample].values, mc.soilmatrix.tr[xsample].values,ksnoise*mc.soilmatrix.ks[xsample].values, mc.soilmatrix.alpha[xsample].values, mc.soilmatrix.n[xsample].values)
                            dD2=vG.dcst_thst(thS.ravel()[idy]/100., mc.soilmatrix.ts[ysample].values, mc.soilmatrix.tr[ysample].values,ksnoise*mc.soilmatrix.ks[ysample].values, mc.soilmatrix.alpha[ysample].values, mc.soilmatrix.n[ysample].values)
                        else:
                            dD1=vG.dcst_thst(thS.ravel()[idx]/100., mc.soilmatrix.ts[xsample].values, mc.soilmatrix.tr[xsample].values,ksnoise[idx]*mc.soilmatrix.ks[xsample].values, mc.soilmatrix.alpha[xsample].values, mc.soilmatrix.n[xsample].values)
                            dD2=vG.dcst_thst(thS.ravel()[idy]/100., mc.soilmatrix.ts[ysample].values, mc.soilmatrix.tr[ysample].values,ksnoise[idy]*mc.soilmatrix.ks[ysample].values, mc.soilmatrix.alpha[ysample].values, mc.soilmatrix.n[ysample].values)
                        dpsi_dtheta=np.sqrt(dD1*dD2)
                        if type(ksnoise)==float:
                            k1=vG.ku_psi(psi1,ksnoise*mc.soilmatrix.ks[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.n[mc.soilgrid.ravel()[idx]-1])
                            k2=vG.ku_psi(psi2,ksnoise*mc.soilmatrix.ks[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.n[mc.soilgrid.ravel()[idy]-1])
                        else:
                            k1=vG.ku_psi(psi1,ksnoise[idx]*mc.soilmatrix.ks[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()[idx]-1],mc.soilmatrix.n[mc.soilgrid.ravel()[idx]-1])
                            k2=vG.ku_psi(psi2,ksnoise[idy]*mc.soilmatrix.ks[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()[idy]-1],mc.soilmatrix.n[mc.soilgrid.ravel()[idy]-1])
                        k=np.sqrt(k1*k2)
                    else:
                        exp_psi=-np.sqrt(mc.psi[thS.ravel()[idx],mc.soilgrid.ravel()[idx]-1]*mc.psi[thS.ravel()[idy],mc.soilgrid.ravel()[idy]-1])
                        dpsi_dtheta=np.sqrt(mc.dpsidtheta[thS.ravel()[idx],mc.soilgrid.ravel()[idx]-1]*mc.dpsidtheta[thS.ravel()[idy],mc.soilgrid.ravel()[idy]-1])
                        k=np.sqrt(mc.ku[thS.ravel()[idx],mc.soilgrid.ravel()[idx]-1]*mc.ku[thS.ravel()[idy],mc.soilgrid.ravel()[idy]-1])
                    
                    #darcy flux into matrix
                    Q=k*-exp_psi/mc.particleD
                    if film:
                        Q[exfilt_retard==1]*=retardfac

                    q_ex=Q*contactfac
                    #exchange impulse
                    p_ex=mc.particleV*(dpsi_dtheta*const.g*1000.)/q_ex

                    #theoretic translatory energy and
                    #structural friction impulse
                    R2=(np.mean(mc.md_area,axis=1)/np.pi)[maccol] #r2 of macropore (mean over depth)
                    u_hag = 1000.*const.g*R2 / (8*0.001308) #mean Hagen Poiseuille laminar flow estimate
                    u_hag *= 2. #max HP flow at center
                    E_tkin = pm*0.5*u_hag**2
                    p_dr = E_tkin/-ux[samplenow] #drag impulse based on current apparent particle velocity

                    ux[samplenow]=-E_tkin/(p_ex+p_dr) #update particle velocity as reduced flow
                    s_red+=ux[samplenow]*t_left #add advective step outside film
                    
                    ##Exfiltration##
                    exfilt[samplenow]=(np.abs(q_ex*t_left)>mc.particleD[0]*0.5)

                #perform advection
                particles_znew[samplenow]+=s_red
                s_red_store[samplenow]=s_red
                
                #check clogging of macropore
                #this may be relevant for cohesive soils with small, coated macropores
                if (clog_switch==True):
                    #functions for clogging calculation
                    #check macropore capacity (clogging)
                    #check if clogging occurs and where
                    def clogpos(idx,idy,idz):
                        x=-1
                        if (idx!=idy):
                            #debug:
                            #print np.shape(mc.maccap), maccol, idz
                            #idz[idz>=np.shape(mc.maccap)[1]]=np.shape(mc.maccap)[1]-1
                            idz_ref=np.where(refpos>idz)[0][0]-1

                            t = mfilling[idx:idy]-mc.maccap[maccol,idz_ref]
                            if any(t<0.):
                                x=idx+np.nonzero(t<0.)[0][0] #mzid of clogged cell
                            else:
                                x=idy
                        return x
                    vclogpos=np.vectorize(clogpos)

                    z_proj=particles_znew[samplenow]
                    macincr=vfindincr(-z_proj)
                    clog=vclogpos(particles_mzid[samplenow],proj_mzid[samplenow],macincr.astype(np.int))

                    #cut advection at clogging
                    cid=(clog>=0)
                    if any(cid):
                        #update z_proj to center of last free cell before clog
                        z_proj[cid]=-mc.particleD*(clog[cid]-0.5)
                        particles_znew[samplenow[cid]]=np.amax([particles_znew[samplenow[cid]],z_proj[cid]],axis=0)


            #set for exfiltration if excceding macropore depth
            exfilt_low = (particles_znew < -mc.md_macdepth[maccol])
            particles_znew[exfilt_low] = -mc.md_macdepth[maccol]

            particles_zid=np.floor(particles_znew/mc.mgrid.vertfac.values).astype(int)
            particles_zid[particles_zid>=mc.mgrid.vertgrid.values[0]]=mc.mgrid.vertgrid.values[0]-1
                
            #assign new z into data frame:
            [lat_new,z_new,nodrain]=boundcheck(particles.lat.iloc[midx],particles_znew,mc)
            particles.loc[particles.flag==(maccol+1),'z']=z_new #particles_znew
            particles.loc[particles.flag==(maccol+1),'cell']=cellgrid(particles.loc[particles.flag==(maccol+1),'lat'].values,particles.loc[particles.flag==(maccol+1),'z'].values,mc).astype(np.int64)
            
            #assign updated advective velocity to particles
            particles.loc[particles.flag==(maccol+1),'advect']=ux #particles_znew

            #debug
            exfilt+=exfilt_low
            if any(exfilt):
                exfilt_p+=sum(exfilt)
                idy=midx[exfilt]
                macincr=np.fmin(vfindincr(particles_znew[exfilt]),np.shape(mc.md_contact)[1]-1)
                particles.flag.iloc[idy]=0
                particles.lat.iloc[idy]=mc.md_pos[maccol]+mc.md_contact[maccol,macincr]*(np.random.rand(sum(exfilt))-0.5)

            #handle draining particles if any
            if any(-nodrain):
                particles.flag.iloc[midx[-nodrain]]=len(mc.maccols)+1
                particles.z.iloc[midx[-nodrain]]=mc.soildepth-0.0001

    return [particles,s_red_store,exfilt_p]



def mx_mp_interact(particles,npart,thS,mc,dt,dynamic_pedo=False,ksnoise=1.):
    '''Calculate if matrix particles infiltrate into a macropore at the inferface areas
    '''
    thS=thS.ravel()
    idx=np.where(thS>mc.FC[mc.soilgrid-1].ravel())[0]
    if len(idx)>0:
        #a) exfiltration into macropores
        idy=mc.macconnect.ravel()[idx]>0
        if any(idy):
            idc=np.in1d(particles.cell.values.astype(np.int),idx[idy]) #get index vector which particles are in affected cells
            #DEBUG: maybe check if any idc is true?
            #we assume diffusive transport into macropore - allow diffusive step and check whether particeD/2 is moved -> then assign to macropore
            N=np.sum(idc)
            xi=np.random.rand(N)
            if dynamic_pedo:
                xsample=mc.soilgrid.ravel()[particles.cell[idc].values]-1
                if type(ksnoise)==float:
                    D=vG.D_thst(thS[particles.cell[idc].values]/100.,mc.soilmatrix.ts[xsample],mc.soilmatrix.tr[xsample],ksnoise*mc.soilmatrix.ks[xsample],mc.soilmatrix.alpha[xsample],mc.soilmatrix.n[xsample]).values
                else:
                    D=vG.D_thst(thS[particles.cell[idc].values]/100.,mc.soilmatrix.ts[xsample],mc.soilmatrix.tr[xsample],ksnoise[particles.cell[idc].values]*mc.soilmatrix.ks[xsample],mc.soilmatrix.alpha[xsample],mc.soilmatrix.n[xsample]).values
            else:
                D=mc.D[thS[particles.cell[idc].values],mc.soilgrid.ravel()[particles.cell[idc].values]-1]
            step_proj=(xi*((2.*D*dt)**0.5))
            ida=(step_proj>=mc.particleD/2.)
            if any(ida):
                particles.loc[idc[ida],'flag']=mc.macconnect.ravel()[particles.cell[idc[ida]].values    ]
        #b) bulk flow advection
        idb=np.in1d(particles.cell.values.astype(np.int),idx.astype(np.int))
        # allow only one particle in appropriate cell to move advectively - DEBUG: this should be an explicit parameter
        dummy, idad = np.unique(particles.cell[idb].values.astype(np.int), return_index=True)
        ida=np.where(idb)[0][idad]
        z_new=particles.z.values[ida]+particles.advect.values[ida]*dt
        z_new[z_new<mc.soildepth]=mc.soildepth+0.0000001
        particles.z.iloc[ida]=z_new
        particles.cell.iloc[ida]=cellgrid(particles.lat.values[ida],particles.z.values[ida],mc).astype(np.int64)

    return particles

def mx_mp_interact_nobulk(particles,npart,thS,mc,dt,dynamic_pedo=False,ksnoise=1.):
    '''Calculate if matrix particles infiltrate into a macropore at the inferface areas
    '''
    thS=thS.ravel()
    idx=np.where(thS>mc.FC[mc.soilgrid-1].ravel())[0]
    if len(idx)>0:
        #flag for exfiltration into adjoined macropores
        thS0=thS*0.
        thS0.ravel()[idx]=mc.macconnect.ravel()[idx]
        idy=np.where(thS0.ravel()>0.)[0]
        if any(idy):
            idc=np.in1d(particles.cell.values.astype(np.int),idy) #get index vector which particles are in affected cells
            #DEBUG: maybe check if any idc is true?
            #we assume diffusive transport into macropore - allow diffusive step and check whether particeD/2 is moved -> then assign to macropore
            N=np.sum(idc)
            xi=np.random.rand(N)
            if dynamic_pedo:
                xsample=mc.soilgrid.ravel()[particles.cell[idc].values]-1
                if type(ksnoise)==float:
                    D=vG.D_thst(thS[particles.cell[idc].values]/100.,mc.soilmatrix.ts[xsample],mc.soilmatrix.tr[xsample],ksnoise*mc.soilmatrix.ks[xsample],mc.soilmatrix.alpha[xsample],mc.soilmatrix.n[xsample]).values
                else:
                    D=vG.D_thst(thS[particles.cell[idc].values]/100.,mc.soilmatrix.ts[xsample].values,mc.soilmatrix.tr[xsample].values,ksnoise[particles.cell[idc].values]*mc.soilmatrix.ks[xsample].values,mc.soilmatrix.alpha[xsample].values,mc.soilmatrix.n[xsample].values)
            else:
                D=mc.D[thS[particles.cell[idc].values],mc.soilgrid.ravel()[particles.cell[idc].values]-1]
            step_proj=(xi*((6*D*dt)**0.5))
            ida=(step_proj>=mc.particleD/2.)
            if any(ida):
                particles.loc[idc[ida],'flag']=mc.macconnect.ravel()[particles.cell[idc[ida]].values]
                particles.loc[idc[ida],'advect']=assignadvect(sum(idc[ida]),mc,mc.macconnect.ravel()[particles.cell[idc[ida]].values])

    return particles

def part_diffusion_split(particles,npart,thS,mc,dt,uffink_corr=True,splitfac=5,vertcalfac=1.,latcalfac=1.,precswitch=True,dynamic_pedo=False,ksnoise=1.):
    '''Calculate Diffusive Particle Movement
       Based on state in grid use diffusivity as foundation of 2D random walk.
       Project step and check for boundary conditions and further restrictions.
       Update particle positions.
    '''
    #N_tot=len(particles.z) #number of particles

    #splitsample particles randomly
    #splitfac=5
    #splitref=np.floor(N_tot/splitfac).astype(int)
    
    #create subsets for splitsampling
    N_tot=particles[particles.flag==0].index
    if (len(N_tot)/splitfac)*splitfac!=len(N_tot):
        adddummy=int(np.round((np.ceil(len(N_tot)/float(splitfac))-len(N_tot)/float(splitfac))*splitfac))
        N_tot=np.concatenate([N_tot,np.zeros(adddummy)*np.nan])
    sampleset=np.random.permutation(N_tot).reshape([splitfac,len(N_tot)/splitfac])
    
    for subsample in np.arange(splitfac):
        #samplenow=sampleset[(subsample*splitref):((subsample+1)*splitref-1)]
        samplenow=sampleset[subsample][sampleset[subsample]>=0]
        N=len(samplenow) #number of particles handled

        # 1D Random Walk function with additional correction term for
        # non-static diffusion after Uffink 1990 p.15 & p.24ff and Kitanidis 1994
        xi=np.random.rand(N,2)*2.-1.
        thSx=thS.ravel()
        if precswitch:
            thSx[thSx>100]=100
        else:
            thSx[thSx>80]=80

        #[D,u,theta]=vG.Dku_thst_f(thSx/100.,mc.soilgrid.ravel()-1,mc)
        #u=u/theta
        #D=D/(theta**2)
        if dynamic_pedo:
            theta=vG.theta_thst(thSx,mc.soilmatrix.ts[mc.soilgrid.ravel()-1],mc.soilmatrix.tr[mc.soilgrid.ravel()-1]).values/100.
            u=vG.ku_thst(thSx/100.,mc.soilmatrix.ks[mc.soilgrid.ravel()-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()-1],mc.soilmatrix.n[mc.soilgrid.ravel()-1]).values/theta
            D=vG.D_thst(thSx/100.,mc.soilmatrix.ts[mc.soilgrid.ravel()-1],mc.soilmatrix.tr[mc.soilgrid.ravel()-1],ksnoise*mc.soilmatrix.ks[mc.soilgrid.ravel()-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()-1],mc.soilmatrix.n[mc.soilgrid.ravel()-1]).values*theta
            #print type(theta)
            #print type(u)
            #print type(D)
        else:
            u=mc.ku[thSx,mc.soilgrid.ravel()-1]/mc.theta[thSx,mc.soilgrid.ravel()-1]
            D=mc.D[thSx,mc.soilgrid.ravel()-1]*mc.theta[thSx,mc.soilgrid.ravel()-1]

        vert_sproj=vertcalfac*(dt*u[particles.cell[samplenow].values.astype(np.int)] + (xi[:,0]*((2*D[particles.cell[samplenow].values.astype(np.int)]*dt)**0.5)))
        lat_sproj=latcalfac*(xi[:,1]*((2*D[particles.cell[samplenow].values.astype(np.int)]*dt)**0.5))
        
        if (uffink_corr==True):
            #Itô Scheme after Uffink 1990 and Kitanidis 1994 for vertical step
            #modified Stratonovich Scheme after Kitanidis 1994 for lateral step
            dx=np.sqrt(vert_sproj**2+lat_sproj**2)
        
            # project step and updated state
            # new positions
            lat_proj=particles.lat
            z_proj=particles.z
            lat_proj[samplenow]=particles.lat[samplenow].values+lat_sproj
            z_proj[samplenow]=particles.z[samplenow].values-vert_sproj
            [lat_proj,z_proj,nodrain]=boundcheck(lat_proj,z_proj,mc)
            [thSx,npartx]=gridupdate_thS(lat_proj,z_proj,mc) 
            thSx=thSx.ravel()
            cell_proj=cellgrid(lat_proj,z_proj,mc).astype(np.int64)
            #cut thSx>1 and thSx<0 for projection - in case they appear
            if precswitch:
                thSx[thSx>100]=100
            else:
                thSx[thSx>80]=80
                thSx[thSx<0]=0
            
            #[D_proj,u_proj,theta_proj]=vG.Dku_thst_f(thSx/100.,mc.soilgrid.ravel()-1,mc)
            #u_proj=u_proj/theta_proj
            #D_proj=D_proj/(theta_proj**2)
            if dynamic_pedo:
                theta_proj=vG.theta_thst(thSx,mc.soilmatrix.ts[mc.soilgrid.ravel()-1],mc.soilmatrix.tr[mc.soilgrid.ravel()-1]).values/100.
                u_proj=vG.ku_thst(thSx/100.,ksnoise*mc.soilmatrix.ks[mc.soilgrid.ravel()-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()-1],mc.soilmatrix.n[mc.soilgrid.ravel()-1]).values/theta_proj
                D_proj=vG.D_thst(thSx/100.,mc.soilmatrix.ts[mc.soilgrid.ravel()-1],mc.soilmatrix.tr[mc.soilgrid.ravel()-1],ksnoise*mc.soilmatrix.ks[mc.soilgrid.ravel()-1],mc.soilmatrix.alpha[mc.soilgrid.ravel()-1],mc.soilmatrix.n[mc.soilgrid.ravel()-1]).values*theta_proj
            else:
                u_proj=mc.ku[thSx,mc.soilgrid.ravel()-1]/mc.theta[thSx,mc.soilgrid.ravel()-1]
                D_proj=mc.D[thSx,mc.soilgrid.ravel()-1]*mc.theta[thSx,mc.soilgrid.ravel()-1]
            
            corrD=np.abs(D_proj[particles.cell[samplenow].values.astype(np.int)]-D[particles.cell[samplenow].values.astype(np.int)])/dx
            corrD[dx==0.]=0.
            D_mean=np.sqrt(D_proj[particles.cell[samplenow].values.astype(np.int)]*D[particles.cell[samplenow].values.astype(np.int)])
            corru=np.sqrt(u[particles.cell[samplenow].values.astype(np.int)]*u_proj[particles.cell[samplenow].values.astype(np.int)])
            #corrD[corrD>corru]=corru[corrD>corru]
            # corrected step
            vert_sproj=vertcalfac*((corru-corrD)*dt + (xi[:,0]*((2*D[particles.cell[samplenow].values.astype(np.int)]*dt)**0.5)))
            lat_sproj=latcalfac*(xi[:,1]/np.abs(xi[:,1]))*corrD*dt + (xi[:,1]*((2*D_mean*dt)**0.5))

        # new positions
        lat_new=particles.lat
        z_new=particles.z
        lat_new[samplenow]=particles.lat[samplenow].values+lat_sproj
        z_new[samplenow]=particles.z[samplenow].values-vert_sproj
        [lat_new,z_new,nodrain]=boundcheck(lat_new,z_new,mc)
        [lat_new,z_new,nodrain2]=boundcheck(lat_new,z_new,mc)

        # saturation check
        [thS,npart]=gridupdate_thS(lat_new[nodrain],z_new[nodrain],mc) #DEBUG: externalise smooth parameter

        if dynamic_pedo:
            phi_mx=vG.psi_thst(thS.ravel()/100.,mc.soilmatrix.alpha[mc.soilgrid.ravel()-1],mc.soilmatrix.n[mc.soilgrid.ravel()-1]).values
        else:
            phi_mx=mc.psi[thS.ravel(),mc.soilgrid.ravel()-1]+mc.mxdepth_cr

        particles['z']=z_new
        particles['lat']=lat_new
        particles['cell']=cellgrid(lat_new,z_new,mc).astype(np.int64)
        if any(-nodrain):
            particles.loc[-nodrain,'flag']=len(mc.maccols)+1

    return [particles,thS,npart,phi_mx]


### THIS IS OLD STUFF DOWN HERE:

def plotparticles(runname,t,particles,npart,mc):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.gridspec as gridspec
    
    f_name=''.join([runname,str(t),'.pdf'])
    pdf_pages = PdfPages(f_name)
    fig=plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,2], height_ratios=[1,5])
    ax1 = plt.subplot(gs[0])
    ax11 = ax1.twinx()
    advect_dummy=np.bincount(np.round(100.0*particles.loc[((particles.age>200.)),'lat'].values).astype(np.int))
    old_dummy=np.bincount(np.round(100.0*particles.loc[((particles.age<200.)),'lat'].values).astype(np.int))
    ax1.plot((np.arange(0,len(advect_dummy))/100.)[1:],advect_dummy[1:],'b-')
    ax11.plot((np.arange(0,len(old_dummy))/100.)[1:],old_dummy[1:],'g-')
    ax11.set_ylabel('Particle Count', color='g')
    ax11.set_xlim([0.,mc.mgrid.width.values])
    ax1.set_xlim([0.,mc.mgrid.width.values])
    ax1.set_ylabel('New Particle Count', color='b')
    ax1.set_xlabel('Lat [m]')
    ax1.set_title('Lateral Particles Concentration')
    
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    ax2.text(0.1, 0.8, 'Particles @ t='+str(t)+'s', fontsize=20)
    
    ax3 = plt.subplot(gs[2])
    plt.imshow(sp.ndimage.filters.median_filter(npart,size=mc.smooth),vmin=1, vmax=mc.part_sizefac, cmap='jet')
    #plt.imshow(npart)
    plt.colorbar()
    plt.xlabel('Width [cells a 5 mm]')
    plt.ylabel('Depth [cells a 5 mm]')
    plt.title('Particle Density')
    plt.tight_layout()

    ax4 = plt.subplot(gs[3])
    ax41 = ax4.twiny()
    z1=np.append(particles.loc[((particles.age>200.)),'z'].values,mc.onepartpercell[1][:mc.mgrid.vertgrid.values.astype(int)])
    advect_dummy=np.bincount(np.round(-100.0*z1).astype(np.int))-1
    old_dummy=np.bincount(np.round(-100.0*particles.loc[((particles.age<200.)),'z'].values).astype(np.int))
    ax4.plot(advect_dummy,(np.arange(0,len(advect_dummy))/-100.),'b-',label='new particles')
    ax41.plot(old_dummy,(np.arange(0,len(old_dummy))/-100.),'g-',label='old particles')
    ax41.set_xlabel('Old Particle Count', color='g')
    ax4.set_xlabel('New Particle Count', color='b')
    ax4.set_ylabel('Depth [m]')
    #ax4.set_title('Number of Particles')
    ax4.set_ylim([mc.mgrid.depth.values,0.])
    ax4.set_xlim([0.,np.max(advect_dummy)])
    ax41.set_xlim([0.,np.max(old_dummy[1:])])
    ax41.set_ylim([mc.mgrid.depth.values,0.])
    #handles1, labels1 = ax4.get_legend_handles_labels() 
    #handles2, labels2 = ax41.get_legend_handles_labels() 
    #ax4.legend(handles1+handles2, labels1+labels2,loc=4)
    #    ax41.legend(loc=4)
    
    plt.show()
    pdf_pages.savefig(fig)
    pdf_pages.close()
    print(''.join(['wrote graphic to ',f_name]))






