# coding=utf-8

import numpy as np
import scipy as sp
import pandas as pd
#import matplotlib.pyplot as plt
import scipy.constants as const
#import scipy.ndimage as spn
#from scipy.ndimage.filters import uniform_filter as unif
#import multiprocessing
#import dataread as dr
import vG_conv as vG
#from numba import double
#from numba.decorators import jit
from macfil_c import macfil_c
from filmflow_c import filmflow_c, macmatrix_c
from binupdate_c import binupdate_c, binupdate_c2
import pickle as pickle
#from filmflow_test import filmflow_t
#from joblib import Parallel, delayed  


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

    return cell.astype(np.intp)


def gridupdate_thS(lat,z,mc,gauss=0.5):
    '''Calculates thetaS from particle density
    '''
    #import numpy as np
    #import scipy as sp
    #import scipy.ndimage as spn
    npart=np.zeros((mc.mgrid.vertgrid.values[0],mc.mgrid.latgrid.values[0]), dtype=np.intp)*(2*mc.part_sizefac)
    lat1=np.append(lat,mc.onepartpercell[0])
    z1=np.append(z,mc.onepartpercell[1])
    cells=cellgrid(lat1,z1,mc)
    #DEBUG:
    cells[cells<0]=0
    trycount = np.bincount(cells)
    trycount=trycount-1 #reduce again by added particles
    npart[np.unravel_index(np.arange(mc.mgrid.cells.astype(np.intp)),(mc.mgrid.vertgrid.values.astype(np.intp)[0],mc.mgrid.latgrid.values.astype(np.intp)[0]))] = trycount
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
            mc.tsref=np.tile(np.append((mc.particleD/circumsegment)[::-1],(mc.particleD/circumsegment)),mc.mgrid.vertgrid[0]).reshape(np.shape(mc.soilgrid))
            #this is the VOLUME of saturation in the respective cell of the half cylinder
        ths_part=mc.soilmatrix.ts[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid))*(2*mc.part_sizefac)
        #thetaS=npart.astype(np.float)*mc.particleV/mc.tsref
        thetaS=mc.tsref*npart.astype(np.float)/ths_part
    elif mc.prects=='column2':
        mc.moistfac=1.
        if mc.colref==False:
            macref=mc.particleD/(0.005*np.pi)
            circumsegment=mc.particleD/((np.arange(mc.mgrid.latgrid.values/2)+1.)*mc.mgrid.latfac.values*np.pi)
            mc.moistfac=np.tile(np.append(circumsegment[::-1],circumsegment),mc.mgrid.vertgrid[0]).reshape(np.shape(mc.soilgrid))
            #this is the VOLUME of saturation in the respective cell of the half cylinder
        ths_part=mc.soilmatrix.ts[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid))*(2*mc.part_sizefac)
            #thetaS=npart.astype(np.float)*mc.particleV/mc.tsref
        thetaS=mc.moistfac*npart.astype(np.float)/ths_part
    else:
        ths_part=mc.soilmatrix.ts[mc.soilgrid.ravel()-1].values.reshape(np.shape(mc.soilgrid))*(2*mc.part_sizefac)
        thetaS=npart.astype(np.float)/ths_part
    thetaS[thetaS>0.99]=0.99
    thetaS[thetaS<0.1]=0.1
    return [(thetaS*100).astype(np.int),npart]

def applyParallel(dfGrouped, func):
    from multiprocessing import Pool, cpu_count
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list)

def binupdate_cx(particles, mc):
    #cython cannot use argsort...
    ksref = mc.soilmatrix.ks.values
    Dref = np.percentile(mc.D, mc.LTEpercentile, axis=0)
    lbin = len(mc.ku)
    splf = mc.splitfac
    dt = mc.dt

    bins = particles.LTEbin.values
    cell = mc.soilgrid.ravel()[particles.cell.iloc[0]]

    [bins, order] = binupdate_c(bins, cell, dt, ksref, Dref, splf, lbin)
    particles.iloc[order, 9] = bins
    return particles

def binupdate_cx2(particles, mc):
    #cython cannot use argsort...
    ksref = mc.soilmatrix.ks.values
    Dref = np.percentile(mc.D, mc.LTEpercentile, axis=0)
    lbin = len(mc.ku)
    splf = mc.splitfac
    dt = mc.dt

    bins = particles.LTEbin.values
    cells = particles.cell.values
    soilid = mc.soilgrid.ravel()[cells].astype(np.int64)
    ncells = mc.mgrid.cells[0]

    [newbins, neworder] = binupdate_c2(bins, cells, soilid, ncells, dt, ksref, Dref, splf, lbin)
    particles.iloc[:, 9] = newbins

    return particles

def binupdate_pdx(bins, pct, dt, ksref, lbin, Dref, spltf):
    #bins = particles.LTEbin.values
    #cell = particles.cell.iloc[0]
    #ksref = mc.soilmatrix.ks[mc.soilgrid.ravel()[cell] - 1]
    #Dref = mc.D[:, mc.soilgrid.ravel()[cell] - 1]
    #lbin = len(mc.ku)

    order = np.argsort(bins)  # current bins in cell
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
        bins_empty = np.append(np.arange(np.intp(np.percentile(bins, pct)), lbin - 2, 1), lbin - 2)
        tmix = ((ksref * dt) ** 2) / Dref[np.intp(np.round(np.amin([lbin - 1, np.percentile(bins_empty, pct)])))]
        step_theoret = np.amax(deviation) * (
                dt / spltf) / tmix  # given tmix as time to reach LTE the current deviation from LTE is projected to be reduced within the current time step by its share
        bins[order[deviation > 0]] = np.intp(
            LTEref[deviation > 0] + np.fmax(deviation[deviation > 0] - step_theoret, 0))

    # this is still an issue but deprecated for now:
    # mc.LTEmemory[cell]+=step_theoret #through the discrete step definition an accumulation is needed. DEBUG: this is a far too simple representation at it is bound to the cell not to the particles
    # DEBUG:
    if any(bins >= lbin):
        bins[bins >= lbin] = np.random.randint(lbin - 1, size=np.sum(bins >= lbin))

    return bins

def binupdate_pd(particles,mc):
    bins=particles.LTEbin.values
    cell=particles.cell.iloc[0]
    
    #debug
    #if not (cell>=0):
    #    print('Cell turned nan')
    #    print(particles)

    #sorted list:
    order=np.argsort(bins) #current bins in cell
    LTEref=np.arange(len(bins)) #bin setting at LTE
    deviation=bins[order]-LTEref #deviation between the two

    #reorganise single step deviations:
    #onestep=np.where(abs(deviation)==1)[0]
    bins[order[abs(deviation)==1]]-=deviation[abs(deviation)==1]

    deviation=bins[order]-LTEref #deviation between the two

    if any(deviation<0): #if there are still particles of smaller pores they are shifted to the enc
        #bins[order[deviation<0]]=np.amax(bins)+np.arange(np.sum(deviation<0))
        bins[order[deviation<0]]=np.intp(LTEref[deviation<0]-deviation[deviation<0])
    
    if any(deviation>0): #if there are still particles of larger pores they mix towards LTE with t_mix
        #allow reorganisation to LTE with tmix=((ks*dt)**2)/Dmix
        #after tmix LTE is achieved this compares to dt and gives the current reorganisation maximum.
        #tmix=((mc.soilmatrix.ks[mc.soilgrid.ravel()[cell]-1]*mc.dt)**2)/mc.D[np.intp(np.round(np.amin([len(mc.D)-1,np.percentile(bins,mc.LTEpercentile)]))),mc.soilgrid.ravel()[cell]-1]

        #alternative: tmix draws from D in empty bins (bin percentile to saturation)
        bins_empty=np.append(np.arange(np.intp(np.percentile(bins,mc.LTEpercentile)),len(mc.ku)-2,1),len(mc.ku)-2)
        tmix=((mc.soilmatrix.ks[mc.soilgrid.ravel()[cell]-1]*mc.dt)**2)/mc.D[np.intp(np.round(np.amin([len(mc.D)-1,np.percentile(bins_empty,mc.LTEpercentile)]))),mc.soilgrid.ravel()[cell]-1]
        step_theoret=np.amax(deviation)*(mc.dt/mc.splitfac)/tmix #given tmix as time to reach LTE the current deviation from LTE is projected to be reduced within the current time step by its share
        bins[order[deviation>0]]=np.intp(LTEref[deviation>0]+np.fmax(deviation[deviation>0]-step_theoret,0))
       
    #this is still an issue but deprecated for now:
    #mc.LTEmemory[cell]+=step_theoret #through the discrete step definition an accumulation is needed. DEBUG: this is a far too simple representation at it is bound to the cell not to the particles
    #DEBUG:
    if any(bins>=len(mc.ku)):
        bins[bins>=len(mc.ku)]=np.random.randint(len(mc.ku)-1,size=np.sum(bins>=len(mc.ku)))

    #particles['LTEbin']=bins[order]
    particles.iloc[order,9]=bins
    return particles

def binupdate_np(particles,mc):
    acells=particles.cell.unique()
    for i in acells:
        bins=particles.loc[particles.cell==i].LTEbin.values
        cell=i
    
        #sorted list:
        order=np.argsort(bins) #current bins in cell
        LTEref=np.arange(len(bins)) #bin setting at LTE
        deviation=bins[order]-LTEref #deviation between the two

        #reorganise single step deviations:
        #onestep=np.where(abs(deviation)==1)[0]
        bins[order[abs(deviation)==1]]-=deviation[abs(deviation)==1]

        deviation=bins[order]-LTEref #deviation between the two

        if any(deviation<0): #if there are still particles of smaller pores they are shifted to the enc
            #bins[order[deviation<0]]=np.amax(bins)+np.arange(np.sum(deviation<0))
            bins[order[deviation<0]]=np.intp(LTEref[deviation<0]-deviation[deviation<0])
    
        if any(deviation>0): #if there are still particles of larger pores they mix towards LTE with t_mix
            #allow reorganisation to LTE with tmix=((ks*dt)**2)/Dmix
            #after tmix LTE is achieved this compares to dt and gives the current reorganisation maximum.
            #tmix=((mc.soilmatrix.ks[mc.soilgrid.ravel()[cell]-1]*mc.dt)**2)/mc.D[np.intp(np.round(np.amin([len(mc.D)-1,np.percentile(bins,mc.LTEpercentile)]))),mc.soilgrid.ravel()[cell]-1]

            #alternative: tmix draws from D in empty bins (bin percentile to saturation)
            bins_empty=np.append(np.arange(np.intp(np.percentile(bins,mc.LTEpercentile)),len(mc.ku)-2,1),len(mc.ku)-2)
            tmix=((mc.soilmatrix.ks[mc.soilgrid.ravel()[cell]-1]*mc.dt)**2)/mc.D[np.intp(np.round(np.amin([len(mc.D)-1,np.percentile(bins_empty,mc.LTEpercentile)]))),mc.soilgrid.ravel()[cell]-1]
            step_theoret=np.amax(deviation)*(mc.dt/mc.splitfac)/tmix #given tmix as time to reach LTE the current deviation from LTE is projected to be reduced within the current time step by its share
            bins[order[deviation>0]]=np.intp(LTEref[deviation>0]+np.fmax(deviation[deviation>0]-step_theoret,0))
       
        #this is still an issue but deprecated for now:
        #mc.LTEmemory[cell]+=step_theoret #through the discrete step definition an accumulation is needed. DEBUG: this is a far too simple representation at it is bound to the cell not to the particles
        #DEBUG:
        if any(bins>=len(mc.ku)):
            bins[bins>=len(mc.ku)]=np.random.randint(len(mc.ku)-1,size=np.sum(bins>=len(mc.ku)))

        #particles['LTEbin']=bins[order]
        particles.loc[particles.cell==i].iloc[order,9]=bins

    return particles


def binupdate_mcore(particles,mc):
    #UNTESTED!!!
    print('trying to use untested multicore routine.')
    from joblib import Parallel, delayed  
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(binupdate_pd)(particles.loc[particles.cell==i],mc) for i in np.arange(mc.mgrid.cells.values))
    
    return pd.concat(results)


def saturationcheck_pd(particles,npart,mc):
    deviation = (mc.mxbin-npart).ravel()
    cell_excess = np.where(deviation<0)[0]
    np.random.shuffle(cell_excess)
    # where deviation < 0 : oversaturation
    # there the excess particles in the highest bins need special treatment
    # cells8 = np.append(np.append([-1,1],([-1,0,1]+np.intp(mc.mgrid.latgrid.values))),([-1,0,1]-np.intp(mc.mgrid.latgrid.values))) #window arround cell
    cells9 = np.append(np.append([-1,0,1],([-1,0,1]+np.intp(mc.mgrid.latgrid.values))),([-1,0,1]-np.intp(mc.mgrid.latgrid.values))) #window arround cell
    cells25 = np.append(np.append(np.append(np.append([-2,-1,1,2],([-2,-1,0,1,2]+np.intp(mc.mgrid.latgrid.values))),([-2,-1,0,1,2]-np.intp(mc.mgrid.latgrid.values))),([-2,-1,0,1,2]-2*np.intp(mc.mgrid.latgrid.values))),([-2,-1,0,1,2]+2*np.intp(mc.mgrid.latgrid.values))) #window arround cell

    for i in cell_excess:
        id_excess = particles.LTEbin[particles.cell==i].nlargest(-deviation[i]).index
        if any((i+cells9)<0):
            #saturation excess at surface
            #excess particles back to surface
            particles.loc[id_excess,'z'] = 0.0001
            continue
        
        cells9i = (i+cells9)[(i+cells9)>=0]
        maccells = list(set(cells9i).intersection(mc.maccon))
        # DEBUG: new version would use np.intersect1d(ar1,ar2)

        if len(maccells)>0:
            #saturation excess into macropores
            particles.loc[id_excess,'flag'] = mc.macconnect.ravel()[maccells[0]]
            particles.loc[id_excess,'advect'] = assignadvect(len(id_excess),mc)
            continue

        cells9i = np.fmin(cells9i,len(deviation)-1) #this is a debug as apparently too large indecees appeared
        if np.sum(deviation[cells9i[deviation[cells9i]>0]])>0:
            # redistribute to cells according to their deviation from saturation
            # this is NOT hypothesising pressure conduction! DEBUG!
            redist = np.amin([-deviation[i],np.sum(deviation[cells9i[deviation[cells9i]>0]])])
            openslots = cells9i[deviation[cells9i]>0].repeat(deviation[cells9i[deviation[cells9i]>0]])
            cell_redist = np.random.choice(openslots,redist)

            particles.loc[id_excess[:redist],'cell'] = cell_redist
            [idz,idx]=np.unravel_index(cell_redist,(mc.mgrid.vertgrid[0].astype(np.intp),mc.mgrid.latgrid[0].astype(np.intp)))
            particles.loc[id_excess[:redist],'z'] = mc.mgrid.vertfac.values*(np.random.random_sample(len(idz))+idz)
            particles.loc[id_excess[:redist],'lat'] = mc.mgrid.latfac.values*(np.random.random_sample(len(idx))+idx)
            if redist==-deviation[i]:
                continue
            deviation[i]+=redist
            id_excess=id_excess[redist:]

        #if now still saturation exists, we need to search in larger space...
        # redistribute to cells according to their deviation from saturation
        # this is NOT hypothesising pressure conduction! DEBUG!
        cells25i = (i+cells25)[(i+cells25)>=0]
        cells25i = cells25i[cells25i<len(deviation)]
        maccells = list(set(cells25i).intersection(mc.maccon))
        # DEBUG: new version would use np.intersect1d(ar1,ar2)

        if len(maccells)>0:
            #saturation excess into macropores
            particles.loc[id_excess,'flag'] = mc.macconnect.ravel()[maccells[0]]
            particles.loc[id_excess,'advect'] = assignadvect(len(id_excess),mc)
            continue

        redist = np.amin([-deviation[i],np.sum(deviation[cells25i[deviation[cells25i]>0]])])
        openslots = cells25i[deviation[cells25i]>0].repeat(deviation[cells25i[deviation[cells25i]>0]])
        if len(openslots)>0: #empty openslots only happen at full saturation. this is not yet properly handled. DEBUG.
            cell_redist = np.random.choice(openslots,redist)

            particles.loc[id_excess[:redist],'cell'] = cell_redist
            [idz,idx]=np.unravel_index(cell_redist,(mc.mgrid.vertgrid[0].astype(np.intp),mc.mgrid.latgrid[0].astype(np.intp)))
            particles.loc[id_excess[:redist],'z'] = mc.mgrid.vertfac.values*(np.random.random_sample(len(idz))+idz)
            particles.loc[id_excess[:redist],'lat'] = mc.mgrid.latfac.values*(np.random.random_sample(len(idx))+idx)
        else:
            #push to next macropore
            idx = np.argmin(np.abs(mc.md_pos[mc.maccols]-particles.loc[id_excess[0],'lat']))
            particles.loc[id_excess,'flag'] = mc.maccols[idx]
            particles.loc[id_excess,'advect'] = assignadvect(len(id_excess),mc)
            # i now assume that there is a macropore and that any free water can be transported there

    return particles

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
    z[z>-0.0000001]=-0.0000001

    #SAFE MODE
    while any(lat<0.0):
        lat[lat<0.0]=mc.mgrid.width[0]+lat[lat<0.0]
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
    p_drain=(z<=mc.soildepth)
    z[p_drain]=mc.soildepth+0.000000000001
    return [lat,z,~p_drain]

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

def filmthick(Psi):
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


def mac_advectionX(particles,mc,thS,dt,clog_switch=False,maccoatscaling=1.,exfilt_method='Ediss',film=True,retardfac=1.,dynamic_pedo=False,ksnoise=1.,fc_check=True,npart=[]):
    '''Calculate Advection in Macropore
       Advective particle movement in macropore with retardation of the advective momentum 
       through drag at interface, plus check for each macropore's capacity and possible overload (clog_switch).

       INPUTS

       DEBUG: ALL FLAGS ARE OVERRUN IN PARALLEL VERSION!

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
    
    #s_red=np.array([])
    exfilt_p=0.
    s_red_store=[]
    refpos=np.round(-mc.macposix/mc.particleD).astype(int) #reference position as macropore index
    
    #debug
    if type(npart)==list:
        print('npart became list')
        print(npart)
    
    def p_adv(maccol):
        return mac_advection_paral(particles,dt,thS,npart,mc,maccol,maccoatscaling,exfilt_method,film,dynamic_pedo,ksnoise,fc_check)

    #def mp_calc_adv(mc.maccols)

    #multicore
    #debug: not running because mc has to be passed as module
    #import multiprocessing
    #from functools import partial
    #func = partial(mac_advection_paral,particles,dt,thS,npart,mc)

    # try:
    #     import multiprocessing
    #     pool = multiprocessing.Pool()
    #     mac_adv_results = pool.map(p_adv,np.arange(len(mc.maccols)))
    # except:
    #     print('multiprocessing failed. looping instead.')
    #     
    # try:
    #     import threading
    #     mac_adv_results = []
    #     for maccol in np.arange(len(mc.maccols)):
    #         m_thread = threading.Thread(target=p_adv, args=(maccol,))
    #         mac_adv_results.append(m_thread)
    #         m_thread.start()
    #print('multithreading failed. using sequential version.')
    mac_adv_results = {}
    for maccol in np.arange(len(mc.maccols)):
        mac_adv_results[maccol] = p_adv(maccol)

    #mac_adv_results = mac_advection_paral(particles,mc,thS,dt,clog_switch,maccoatscaling,exfilt_method,film,retardfac,dynamic_pedo,ksnoise,fc_check,npart)

    #[particle_idx_processed,particles_znew,s_red,ux,exfilt_p,mc.mactopfill[maccol],maccol,filmloc,exfilt_mem,exfilt,excess_ix]
    particles_znew=np.array([])
    s_red=np.array([])
    ux=np.array([])
    particle_idx_processed=np.array([])
    #exfilt_p=np.array([])
    filmloc=np.array([])
    exfilt_mem=np.array([])
    exfilt=np.array([[],[],[]])
    excess_ix=np.array([])

    for nres in np.arange(len(mac_adv_results)):
        results = mac_adv_results[nres]
        if type(results)==np.int64:
            print('Parallel macropore computation error.')
            print('Result was: ', results)
            print('Result dictionary:',mac_adv_results)

            xdump = pickle.dumps([results,mac_adv_results])
            f = open('ydump.pick', "xb")
            pickle.dump(xdump, f, protocol=2)
            f.close()
            print('Data stored in ydump.pick')
            break

        else:
            mc.mactopfill[results[6]]=results[5]
            particles_znew=np.append(particles_znew,results[1])
            s_red=np.append(s_red,results[2])
            ux=np.append(ux,results[3])

            #debug:
            ux[ux==0] = -0.0676
            
            #exfilt_p=np.append(exfilt_p,results[4])
            particle_idx_processed=np.append(particle_idx_processed,results[0])
            filmloc=np.append(filmloc,results[7])
            exfilt_mem=np.append(exfilt_mem,results[8])
            if np.shape(results[9])!=(3,0):
                exfilt=np.concatenate((exfilt.T,results[9].T)).T
            excess_ix=np.append(excess_ix,results[10])

    #set for exfiltration if excceding macropore depth
    particle_idx_processed = particle_idx_processed.astype(np.int64)
    exfilt_low = particle_idx_processed[(particles_znew < -mc.md_macdepth[0])]    
    particles_zid=np.floor(particles_znew/mc.mgrid.vertfac.values).astype(int)
    particles_zid[particles_zid>=mc.mgrid.vertgrid.values[0]]=mc.mgrid.vertgrid.values[0]-1

    #assign new z into data frame:
    [lat_new,z_new,nodrain]=boundcheck(particles.lat.loc[particle_idx_processed],particles_znew,mc)
    particles.loc[particle_idx_processed,'z']=z_new #particles_znew
    particles.loc[particle_idx_processed,'cell']=cellgrid(particles.loc[particle_idx_processed,'lat'].values,particles.loc[particle_idx_processed,'z'].values,mc).astype(np.intp)
    #assign updated advective velocity to particles
    particles.loc[particle_idx_processed,'advect']=ux
    particles.loc[particle_idx_processed,'fastlane']=filmloc 
    particles.loc[particle_idx_processed,'exfilt']=exfilt_mem 

    #assign values to exfiltrating particles
    
    exfilt_p = 0.
    if np.shape(exfilt)!=(3,0):
        exfiltx = np.unique(np.append(exfilt[0, :], exfilt_low)).astype(np.int64)
        exfilt_p+=len(exfiltx)
        particles.loc[exfiltx,'flag']=0
        particles.loc[exfiltx,'fastlane']=0
        particles.loc[exfiltx,'exfilt']=0
        particles.loc[exfiltx,'LTEbin']=exfilt[2, :].astype(np.int64) #assign LTE bin to exfilt particle
        particles.loc[exfiltx,'lat']=exfilt[1, :]

    #handle draining particles if any
    if any(~nodrain):
        particles.flag.loc[midx[~nodrain]]=len(mc.maccols)+1
        particles.z.loc[midx[~nodrain]]=mc.soildepth-0.0001
    
    #redistribute excess particles from macropore to surface
    if len(excess_ix)>0:
        particles.loc[excess_ix,'flag']=0 #set back to matrix domain
        particles.loc[excess_ix,'z']=0. #set z to zero > will be passed back to infilt routine

    return [particles,s_red,exfilt_p,mc]

#helper functions
def macpos(z,slotscale,mxgridcell):
    #get index position in macropore
    # idm=np.zeros(len(z)).astype(np.intp)
    # for i in np.arange(len(z)):
    #     dummy=np.where(z[i]>macdepth)[0]
    #     if len(dummy)>1:
    #         idm[i]=int(dummy[0])
    #     elif len(dummy)==0:
    #         idm[i]=mxgridcell-1
    #     else:
    #         idm[i]=int(dummy)
    # return idm
    idm = np.floor(-z/slotscale).astype(int)
    idm = np.fmin(np.fmax(idm,0),mxgridcell)
    return idm

# def macfil(p_mzid,mxgridcell):
#     #get macropore filling and position in film
#     p_mzid[p_mzid<0]=0 #if particles get erroneous positions due to redistribution it is set to zero
#     p_mzid_plus1=np.append(p_mzid,np.arange(mxgridcell))
#     mfilling=np.bincount(p_mzid_plus1)-1
#     #try/except debugging mode:
#     #try:
#     #    mfilling=np.bincount(p_mzid_plus1)-1
#     #except ValueError:
#     #    print(p_mzid,mxgridcell)
#     filmloc=np.ones(len(p_mzid),dtype=int)
#     for idx in np.where(mfilling>0)[0]:
#         idy=np.where(p_mzid==idx)
#         filmloc[idy]=np.arange(mfilling[idx],dtype=int)+1
#     #Outputs: 1 filling state, 2 location in film/distance to porewall
#     return [mfilling, filmloc]

def mac_advection_paral(particles,dt,thS,npart,mc,maccol,maccoatscaling,exfilt_method,film,dynamic_pedo,ksnoise,fc_check):
    macstate=[]
    filmloc=[]
    exfilt_p=0.
    exfilt_tb=np.array([[], [], []])
    exfilt_mem=[]
    particles_znew=[]
    midx=np.array([])
    excess_id=[]
    retardfac=1.
    pm=mc.particlemass/1000. #particle mass conversion into kg
    
    if not particles.loc[particles.flag==(maccol+1)].empty:
        
        particle_idx_processed=particles.loc[particles.flag==(maccol+1)].index

        #old z
        old_z=particles.loc[particles.flag==(maccol+1),'z'].values
        particles_znew=particles.loc[particles.flag==(maccol+1),'z'].values
        
        #project advection step
        midx=particles.index[particles.flag==(maccol+1)]
        
        #DEBUG TODO:
        #move the preamble to the preprocessing

        #id of matrix
        z_cells=np.intp(np.ceil(mc.md_macdepth[maccol]/-mc.mgrid.vertfac.values))
        mac_connect=cellgrid(mc.md_pos[maccol].repeat(z_cells),(np.arange(z_cells)+0.5)*mc.mgrid.vertfac.values,mc)
        mac_connect=mac_connect[mac_connect<mc.mgrid.cells.values]
        z_cells=len(mac_connect)

        #state along macropore
        mac_psi=mc.psi100[sp.ndimage.filters.median_filter(thS,size=mc.smooth).ravel()[mac_connect],mc.soilgrid.ravel()[mac_connect]-1]
        
        #film slots
        #film thick estimate deprecated as it ranges at about one or less particle!

        #thick=filmthick(mac_psi*98.0638) #convert m water column into hPa
        #filmvolume=-mc.mgrid.vertfac.values*thick*mc.mac_contact[:len(thick),maccol]
        ##slotspergrid=np.ceil(filmvolume/mc.particleV).astype(np.intp) #DEBUG: large particles will lead to an overestimation of the film
        ##assume reference film thickness at FC
        #slotspergrid=np.fmax(1,np.round(filmthick(-330)*-mc.mgrid.vertfac.values*mc.mac_contact[:len(mac_connect),maccol]/mc.particleV).astype(np.int))
        #filmlayers=np.ceil(filmvolume/(slotspergrid*mc.particleV)).astype(np.intp)
        #macgrid=np.repeat(mac_connect,slotspergrid)
        #macdepth=np.cumsum(np.repeat(mc.mgrid.vertfac.values*np.ones(len(slotspergrid))/slotspergrid.astype(np.float),slotspergrid))
        #mxgridcell=len(macgrid)-1
        
        #ratio of latgrid to macropore radius to total gridspace
        mac_grid_ratio=np.mean(np.sqrt(mc.md_area)[:,0])/mc.mgrid.latfac.values
        #max particle in macropore reference estimated from column of soil diveded by capacity gives number of slots
        mxgridcell = int(np.ceil(mac_grid_ratio*(mc.md_macdepth[maccol]/-mc.mgrid.vertfac.values)*mc.part_sizefac/mc.maccap[maccol,0]))
        slotscale = mc.md_macdepth[maccol]/mxgridcell

        #id of macropore in soil grid
        mac_cell=np.repeat(mac_connect,np.diff(np.append(np.array(0.),np.round(np.cumsum(np.repeat(-mc.mgrid.vertfac.values/slotscale,z_cells))))).astype(int))[:mxgridcell+1]
        mxgridcell=len(mac_cell)-1
        macdepth=np.arange(mxgridcell)*-slotscale
        #DEBUG:
        #ADD HP FOR FILM. ONLY AT OUTER FILM LAYER THE PARTICLES ARE ALLOWED TO DRAIN.

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

        #maxfill
        dummy_i=np.ceil(mc.md_depth[1:]/slotscale).astype(int)
        dummy_i[1:]=dummy_i[1:]-dummy_i[:-1]
        maxfill=np.repeat(mc.maccap[maccol,:],dummy_i).astype(np.intp)[:mxgridcell]
        
        if any((maxfill==0) & (len(maxfill[maxfill>0])>0)):
            maxfill[maxfill==0]=maxfill[maxfill>0][-1]

        #id in macropore
        particles_mzid=macpos(particles_znew,slotscale,mxgridcell)
        #[mfilling, filmloc]=macfil(particles_mzid,mxgridcell)
        [mfilling, filmloc]=macfil_c(particles_mzid,np.arange(len(particles_mzid)),len(particles_mzid),mxgridcell)

        #debug: print(mfilling[:20])

        #debug: film location changes every time the function is called. this is maybe a bad idea once the flow is dependent on it 
        
        #advective velocity
        ux=particles.loc[particles.flag==(maccol+1),'advect'].values
        #particles reset advective velocity when far from pore wall
        ux[filmloc>1]=assignadvect(sum(filmloc>1),mc)

        if any(ux==0):
            print('ux went zero.')

        s_proj=ux*dt #project step
        z_proj=particles.loc[particles.flag==(maccol+1),'z'].values+s_proj #project new position

        #check for excess particles
        if len(maxfill)!=len(mfilling):
            mfilling=mfilling[:len(maxfill)]

        openslots=maxfill-mfilling
        if any(openslots<0):
            excess=np.where(openslots<0)[0]

            for iex in excess[::-1]:
                iexx=np.where(particles_mzid==iex)[0][:-openslots[iex]]
                #try redistribute above if not top cell:
                if iex>0:
                    while len(iexx)>0:
                        particles_mzid[iexx]-=np.fmin(particles_mzid[iexx],1) #shift 1 or zero up
                        z_proj[iexx]=np.fmin(z_proj[iexx]-slotscale, -0.00001) #add displacement to projected step length
                        particles_znew[iexx]=np.fmin(particles_znew[iexx]-slotscale, -0.00001) #add displacement to projected new position
                        #ux[iexx]=0.
                        if len(iexx)>openslots[iex-1]:
                            iexx=iexx[openslots[iex-1]:]
                            openslots[iex-1]=0
                            iex-=1
                            if iex<1:
                                #ux[iexx]=0.
                                z_proj[iexx]=0.
                                particles_znew[iexx]=0.
                                iexx=[]
                                continue
                        else:
                            openslots[iex-1]-=len(iexx)
                            iexx=[]
                            continue

        #[mfilling, filmloc]=macfil(particles_mzid,mxgridcell)
        [mfilling, filmloc]=macfil_c(particles_mzid,np.arange(len(particles_mzid)),len(particles_mzid),mxgridcell)
        
        if len(maxfill)!=len(mfilling):
            mfilling=mfilling[:len(maxfill)]
        
        openslots=maxfill-mfilling

        #check lower boundary
        nodrain=(z_proj>=mc.soildepth)
        if any(~nodrain):
            z_proj[~nodrain]=mc.soildepth
        #cell of projected step
        proj_mzid=macpos(z_proj,slotscale,mxgridcell)

        exfilt=particles_mzid<0 #create exfilt array with all False
        s_red_store=np.zeros(len(particles_mzid))
        
        #sampleset=np.arange(len(particles_mzid),dtype=np.intp)
        #samplenow=sampleset
        #samplenow=samplenow[samplenow>=0]
        
        #update gridfill
        #particles_mzid=macpos(particles_znew)
        #[mfilling, filmloc]=macfil(particles_mzid,mxgridcell)

        #functions to check free slots in macropore along path
        #mfilling must be defined before as filling state of macropore, thus new def of functions here
        def contactcount(idx,idy):
            #ocuppied slots on course
            return np.count_nonzero(mfilling[idx:idy]==0)
        vcontactcount=np.vectorize(contactcount)
        # def freecount(idx,idy):
        #     #free slots on course
        #     return sum(mfilling[idx:idy]==0)
        # vfreecount=np.vectorize(freecount)

        #soil cell id of start and end
        idx=mac_cell[particles_mzid]
        idy=mac_cell[proj_mzid]
        #s_red=np.zeros(len(samplenow))
        #t_left=np.ones(len(samplenow))*dt
        #u_hag=s_red #advective velocity after hagen-poiseuille
        contactfac=np.ones(len(particles_mzid),dtype=np.float64)

        #project diffusion into matrix
        ##CONTACT FACE##
        if film:
            #assume film initialisation at pore wall 
            exfilt_retard=np.zeros(len(particles_mzid),dtype=int)
            
            #particles will proceed with v_adv to the end of the film
            #therefore the reference will shift to the first free slot
            ib=filmloc>1
            if any(ib) & any((proj_mzid-particles_mzid)>0):
                s_samp=np.where(ib)[0]
                s_samp=s_samp[s_samp<mxgridcell]
                #call cythonised filmflow
                try:
                    [particles_mzid,proj_mzid,mfilling,s_red,t_left] = filmflow_c(particles_mzid, proj_mzid, ux, old_z, macdepth, s_samp, mfilling, len(particles_mzid), mxgridcell, dt)
                except:
                    xdump = pickle.dumps([particles_mzid, proj_mzid, ux, old_z, macdepth, s_samp, mfilling, len(particles_mzid), mxgridcell, dt])
                    f = open('xdump.pick',"xb")
                    pickle.dump(xdump, f, protocol=2)
                    f.close()
                    print('filmflow_c failed. data stored in xdump.pick')
                    [particles_mzid, proj_mzid, mfilling, s_red, t_left] = filmflow_c(particles_mzid, proj_mzid, ux,
                                                                                      old_z, macdepth, s_samp, mfilling,
                                                                                      len(particles_mzid), mxgridcell, dt)

                #[particles_mzid, proj_mzid, mfilling, s_red, t_left] = filmflow_t(particles_mzid, proj_mzid, ux, old_z,macdepth, s_samp, mfilling,len(samplenow), dt)

                idx=mac_cell[particles_mzid] #update reference to soil
                idy=mac_cell[proj_mzid]
            else:
                s_red=np.zeros(len(particles_mzid))
                t_left=np.ones(len(particles_mzid))*dt
                
            filmweight=vcontactcount(particles_mzid,proj_mzid).astype(np.float64) #free slots on course
            passage=(proj_mzid-particles_mzid).astype(np.float64) #length of projected voyage
            
            ia=filmweight>0.
            ic=passage>0.
            if any(ia & ic):
                contactfac[ia & ic]=filmweight[ia & ic]/passage[ia & ic]

            #particles at position 1 in film can be retarded for exfiltration to simulate a film
            exfilt_retard[filmloc==1]=1
            
        else:
            #assume only film particles to interact with the matrix
            dragweight=vcontactcount(particles_mzid,proj_mzid) #free slots on course
            passage=(proj_mzid-particles_mzid).astype(np.float64) #length of projected voyage
            ia=dragweight>0.
            ib=passage>0.
            if any(ia & ib):
                contactfac[ia & ib]=1.-(dragweight[ia & ib]/passage[ia & ib])
            
        #scale contactfac with coating factor
        contactfac/=maccoatscaling

        #split method >FC gets also RWdiff...
        #experienced psi
        xsample=mc.soilgrid.ravel()[idx]-1
        ysample=mc.soilgrid.ravel()[idy]-1
        if dynamic_pedo:
            #WARNING: THIS STRECHES ACROSS MULTIPLE CELLS. REVISE AS BELOW. 
            psi1=vG.psi_thst(thS.ravel()[idx]/100.,mc.soilmatrix.alpha[xsample].values,mc.soilmatrix.n[xsample].values)
            psi2=vG.psi_thst(thS.ravel()[idy]/100.,mc.soilmatrix.alpha[ysample].values,mc.soilmatrix.n[ysample].values)
            exp_psi=-np.sqrt(psi1*psi2)
            if type(ksnoise)==float:
                dD1=vG.dcst_thst(thS.ravel()[idx]/100., mc.soilmatrix.ts[xsample].values, mc.soilmatrix.tr[xsample].values,ksnoise*mc.soilmatrix.ks[xsample].values, mc.soilmatrix.alpha[xsample].values, mc.soilmatrix.n[xsample].values)
                dD2=vG.dcst_thst(thS.ravel()[idy]/100., mc.soilmatrix.ts[ysample].values, mc.soilmatrix.tr[ysample].values,ksnoise*mc.soilmatrix.ks[ysample].values, mc.soilmatrix.alpha[ysample].values, mc.soilmatrix.n[ysample].values)
            else:
                dD1=vG.dcst_thst(thS.ravel()[idx]/100., mc.soilmatrix.ts[xsample].values, mc.soilmatrix.tr[xsample].values,ksnoise[idx]*mc.soilmatrix.ks[xsample].values, mc.soilmatrix.alpha[xsample].values, mc.soilmatrix.n[xsample].values)
                dD2=vG.dcst_thst(thS.ravel()[idy]/100., mc.soilmatrix.ts[ysample].values, mc.soilmatrix.tr[ysample].values,ksnoise[idy]*mc.soilmatrix.ks[ysample].values, mc.soilmatrix.alpha[ysample].values, mc.soilmatrix.n[ysample].values)
            dpsi_dtheta=np.sqrt(dD1*dD2)
            if type(ksnoise)==float:
                k1=vG.ku_psi(psi1,ksnoise*mc.soilmatrix.ks[xsample],mc.soilmatrix.alpha[xsample],mc.soilmatrix.n[xsample])
                k2=vG.ku_psi(psi2,ksnoise*mc.soilmatrix.ks[ysample],mc.soilmatrix.alpha[ysample],mc.soilmatrix.n[ysample])
            else:
                k1=vG.ku_psi(psi1,ksnoise[idx]*mc.soilmatrix.ks[xsample],mc.soilmatrix.alpha[xsample],mc.soilmatrix.n[xsample])
                k2=vG.ku_psi(psi2,ksnoise[idy]*mc.soilmatrix.ks[ysample],mc.soilmatrix.alpha[ysample],mc.soilmatrix.n[ysample])
            k=np.sqrt(k1.values*k2.values)
        else:
            gridspan=np.floor(s_proj/mc.mgrid.vertfac.values).astype(np.intp) #number of grids the step spanns
            mac_u=np.unique(mac_cell)
            #gridspan=np.fmax(0,np.fmin(len(mac_u)-(1+idx),gridspan))
            ssample=mc.soilgrid.ravel()[mac_u]-1
            sxsample=mc.soilgrid.ravel()[mac_connect]-1
            psi_mac=mc.psi100[thS.ravel()[mac_u],ssample]
            dpsi_dtheta_mac=mc.dpsidtheta100[thS.ravel()[mac_u],ssample]
            k_mac=mc.ku100[thS.ravel()[mac_u],ssample]
            D_mac=mc.D100[thS.ravel()[mac_u],ssample]
            dtheta_mac=mc.theta100[100-thS.ravel()[mac_u],ssample]
            FC_excess=(npart.ravel()[mac_u]-mc.FC[ssample])>=0
            FC_free_prob=(npart.ravel()[mac_u]+((mc.gridcellA/mc.particleA).values*mc.soilmatrix.ts)[ssample].values)/((mc.gridcellA/mc.particleA).values*mc.soilmatrix.ts)[ssample].values
            
            idx0=np.ceil(old_z/mc.mgrid.vertfac.values).astype(int)
            idx1=np.fmax(1,np.fmin(gridspan,np.intp(mc.mgrid.vertgrid.values/10))) #maximum 1/10 of cells allowed to check for states along passage
            idx0[idx0>=(len(psi_mac)-1)]=len(psi_mac)-2
            idx1=np.fmin(idx1,len(psi_mac)-1-idx0)
            stretch=np.transpose([idx0,idx1]) 
            
            [Qx, dpsi_dtheta, exp_psi, thick, k, D, thetax] = macmatrix_c(psi_mac, dtheta_mac, dpsi_dtheta_mac, k_mac, D_mac, FC_excess.astype(int), stretch, len(idx), mc.particleD[0])

        Q=Qx
        
        #project Q as random walk velociy without dt based on conditions at interface
        #Q=np.random.rand(len(idx))*((6*D*thetax)**0.5)

        #Q=k*-exp_psi/mc.mgrid.latfac.values
        if film:
            #Q[exfilt_retard==1]*=retardfac
            Q[exfilt_retard==0]/=filmloc[exfilt_retard==0]**2
        q_ex=Q*contactfac

        #exchange impulse
        p_ex=mc.particleV*(dpsi_dtheta*const.g*1000.)/q_ex
        #theoretic translatory energy and
        #structural friction impulse
        R2=(np.mean(mc.md_area,axis=1)/np.pi)[maccol] #r2 of macropore (mean over depth)
        u_hag = 1000.*const.g*R2 / (8*0.001308) #mean Hagen Poiseuille laminar flow estimate
        #u_hag *= 2. #max HP flow at center
        u_hag *= 2.*filmloc/np.max([2.,np.max(filmloc)]) #alternative with given filmloc
        E_tkin = pm*0.5*u_hag**2
        p_dr = E_tkin/-ux #drag impulse based on current apparent particle velocity
        #p_ex[exfilt_retard==0] = 0. #no drag at higher layers
        p_dr /= filmloc**2 #quadratic reduction of drag in film
        p_ex /= filmloc**2 #quadratic reduction of drag in film
        #ux[samplenow]=-E_tkin/((p_ex+p_dr)/filmloc**2) #update particle velocity as reduced flow
        ux=-E_tkin/(p_ex+p_dr) #update particle velocity as reduced flow
        s_red+=ux*t_left #add advective step outside film
        dummy_ux=ux
        
        ##Exfiltration##
        exfilt_mem=particles.loc[particles.flag==(maccol+1),'exfilt'].values.astype('float64')
        exfilt_mem+=np.abs(q_ex*t_left)
        #exfilt[samplenow]=(exfilt_mem>mc.particleD[0]*0.5)
        #exfilt[samplenow]=(exfilt_mem>mc.particleD[0]*0.01) #assume much less entry. DEBUG!
        exfilt=(thick-exfilt_mem)<=0. #any exfilt length exceeding the film layer depth

        if (exfilt_method=='RWdiff') and any(abs(exp_psi)<0.33):
            #which experienced psi is greater than FC
            id_FC=(abs(exp_psi)<0.33)
            #here the RW approach takes place too.
            
            xi=np.random.rand(sum(id_FC))
            #diffusion over projected passage as geo mean of start and end
            #if dynamic_pedo:
            #    psi1=vG.psi_thst(thS.ravel()[idx[id_FC]],mc.soilmatrix.alpha[xsample[id_FC]],mc.soilmatrix.n[xsample[id_FC]]).values
            #    psi2=vG.psi_thst(thS.ravel()[idy[id_FC]],mc.soilmatrix.alpha[ysample[id_FC]],mc.soilmatrix.n[ysample[id_FC]]).values
            #    if type(ksnoise)==float:
            #        D1=vG.D_psi(psi1,ksnoise*mc.soilmatrix.ks[xsample[id_FC]],mc.soilmatrix.ts[xsample[id_FC]],mc.soilmatrix.tr[xsample[id_FC]],mc.soilmatrix.alpha[xsample[id_FC]],mc.soilmatrix.n[xsample[id_FC]])
            #        D2=vG.D_psi(psi2,ksnoise*mc.soilmatrix.ks[ysample[id_FC]],mc.soilmatrix.ts[ysample[id_FC]],mc.soilmatrix.tr[ysample[id_FC]],mc.soilmatrix.alpha[ysample[id_FC]],mc.soilmatrix.n[ysample[id_FC]])
            #    else:
            #        D1=vG.D_psi(psi1,ksnoise[idx[id_FC]]*mc.soilmatrix.ks[xsample[id_FC]],mc.soilmatrix.ts[xsample[id_FC]],mc.soilmatrix.tr[xsample[id_FC]],mc.soilmatrix.alpha[xsample[id_FC]],mc.soilmatrix.n[xsample[id_FC]])
            #        D2=vG.D_psi(psi2,ksnoise[idy[id_FC]]*mc.soilmatrix.ks[ysample[id_FC]],mc.soilmatrix.ts[ysample[id_FC]],mc.soilmatrix.tr[ysample[id_FC]],mc.soilmatrix.alpha[ysample[id_FC]],mc.soilmatrix.n[ysample[id_FC]])
            #    D=np.sqrt(D1*D2)
            #else:
                #D=np.sqrt(mc.D100[thS.ravel()[idx[id_FC]],xsample[id_FC]]*mc.D100[thS.ravel()[idy[id_FC]],ysample[id_FC]])
                #deprecates as calulated 45 lines before

            diff_proj=(xi*((6*D[id_FC]*t_left[id_FC])**0.5))*contactfac[id_FC]

            if film:
                diff_proj[exfilt_retard[id_FC]==1]*=retardfac
            adv_retard=(mc.particleD.repeat(sum(id_FC))-diff_proj)/mc.particleD
            adv_retard[adv_retard<0.]=0.
            
            ux[id_FC]*=adv_retard
            s_red[id_FC]-=np.fmax(0.,dummy_ux[id_FC]-ux[id_FC])*t_left[id_FC]
            
            exfilt[id_FC]=(exfilt[id_FC] | (adv_retard<=0.3))                  

        if any(exfilt):
            idy = midx[exfilt]
            particles_mzid_ex = macpos(particles.z.loc[idy].values,slotscale,mxgridcell)
            thS_ex = thS.ravel()[mac_cell[particles_mzid_ex]]
            idy = idy[thS_ex < 99]
            macincr = np.fmin(vfindincr(particles_znew[exfilt]), np.shape(mc.md_contact)[1] - 1)
            macincr = macincr[thS_ex < 99]
            lat_y=mc.md_pos[maccol]+mc.md_contact[maccol,macincr]*(np.random.random_sample(len(idy))-0.5)
            LTE_y=np.intp((np.amax(mc.mxbin)-2)*(thS_ex[thS_ex<99]/100.))
            exfilt_tb = np.zeros([3, len(idy)])
            exfilt_tb[0, :] = idy
            exfilt_tb[1, :] = lat_y
            exfilt_tb[2, :] = LTE_y

        ##CHECK MACROPORE CAPACITY##
        particles_mzid_proj=macpos(particles_znew+s_red,slotscale,mxgridcell)
        #[mfilling_proj, filmloc_p]=macfil(particles_mzid_proj,mxgridcell)
        [mfilling_proj, filmloc_p]=macfil_c(particles_mzid,np.arange(len(particles_mzid)),len(particles_mzid),mxgridcell)
        #DEBUG
        mfilling_proj=mfilling_proj[:len(maxfill)]
        openslots_proj=maxfill-mfilling_proj
        excess_id=[]
        if any(openslots_proj<0):
            excess=np.where(openslots_proj<0)[0]

            for iex in excess[::-1]:
                iexx=np.where(particles_mzid_proj==iex)[0][openslots_proj[iex]:]
                #redistribute above:
                while len(iexx)>0:
                    #s_red[iexx]+=np.fmin(-s_red[iexx],mc.particleD)
                    s_red[iexx]+=slotscale
                    openslots_proj[iex]=np.amax([0,openslots_proj[iex]-len(iexx)])
                    if len(iexx)>openslots_proj[iex-1]:
                        iexx=iexx[openslots_proj[iex-1]:]
                        iex-=1
                        if iex<1:
                            s_red[iexx]=0.
                            particles_znew[iexx]=0.
                            openslots_proj[0]=0
                            excess_id=iexx
                            iexx=[]
                            continue
                    else:
                        iexx=[]
                        continue

        if len(excess_id)>0:
            mc.mactopfill[maccol]=maxfill[0]
        else:
            mc.mactopfill[maccol]=maxfill[0]-openslots_proj[0]
        #piling of any particles approaching this spot (like clogging)
        #positive psi for all particles in cue as h above clogged particle
        # mc.maccap is now better defined

        #perform advection
        particles_znew+=s_red

    else:
        particle_idx_processed=[]
        s_red=[]
        ux=[]
        exfilt=[]
        mc.mactopfill[maccol]=mc.maccap[maccol,0]

    return [particle_idx_processed,particles_znew,s_red,ux,exfilt_p,mc.mactopfill[maccol],maccol,filmloc,exfilt_mem,exfilt_tb,midx[excess_id]]


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
                D=mc.D100[thS[particles.cell[idc].values],mc.soilgrid.ravel()[particles.cell[idc].values]-1]
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
        particles.cell.iloc[ida]=cellgrid(particles.lat.values[ida],particles.z.values[ida],mc).astype(np.intp)

    return particles

def mx_mp_interact_nobulk(particles,npart,thS,mc,dt,dynamic_pedo=False,ksnoise=1.):
    '''Calculate if matrix particles infiltrate into a macropore at the inferface areas
    '''
    thS=thS.ravel()
    npart_r=npart.ravel()
    #idx=np.where(thS>mc.FC[mc.soilgrid-1].ravel())[0]
    idx=np.where(npart_r>mc.FC[mc.soilgrid-1].ravel())[0]
    if len(idx)>0:
        #flag for exfiltration into adjoined macropores
        thS0=npart_r*0.
        thS0[idx]=mc.macconnect.ravel()[idx]
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
                D=mc.D100[thS[particles.cell[idc].values],mc.soilgrid.ravel()[particles.cell[idc].values]-1]
            step_proj=(xi*((6*D*dt)**0.5))
            ida=(step_proj>=mc.particleD/2.)
            if any(ida):
                particles.loc[particles.index[idc][ida],'flag']=mc.macconnect.ravel()[particles.loc[particles.index[idc][ida],'cell'].values]
                particles.loc[particles.index[idc][ida],'advect']=assignadvect(sum(ida),mc,mc.macconnect.ravel()[particles.loc[particles.index[idc][ida],'cell'].values])

    return particles


def part_diffusion_binned_pd(particles,npart,thS,mc,panda=True):
    #create subsets for splitsampling
    N_id=particles[particles.flag==0].index.values
    N_tot=np.arange(len(N_id))
    if np.ceil((float(len(N_tot))/mc.splitfac))*mc.splitfac!=len(N_tot):
        adddummy=int(np.ceil((float(len(N_tot))/mc.splitfac))*mc.splitfac-len(N_tot))
        N_tot=np.concatenate([N_tot,np.zeros(adddummy)*np.nan])
    sampleset=np.random.permutation(N_tot).reshape([mc.splitfac,int(np.ceil((float(len(N_tot))/mc.splitfac)))]).astype(np.intp)

    for subsample in np.arange(mc.splitfac):
        S_id=N_id[sampleset[subsample][sampleset[subsample]>=0].astype(np.intp)]
        if not all(S_id>=0):
            print('PROBLEM at S_id')
            print(S_id[S_id>=0])
        par_diff_sub_pd(S_id,particles,thS,mc,npart)

    #update states 
    if panda=='multi': 
        particles=binupdate_mcore(particles,mc)
    if panda==False:
        particles=binupdate_np(particles,mc)
    else: #(pandas groupby)
        #particles=particles.groupby('cell').apply(binupdate_pd,mc)
        particles=binupdate_cx2(particles, mc)

    [thS,npart] = gridupdate_thS(particles.loc[particles.flag==0,'lat'].values,particles.loc[particles.flag==0,'z'].values,mc)
    particles = saturationcheck_pd(particles,npart,mc)
    phi_mx=mc.psi100[thS.ravel(),mc.soilgrid.ravel()-1]+mc.mxdepth_cr
    
    return [particles,thS,npart,phi_mx]


def par_diff_sub_pd(S_id,particles,thS,mc,npart):
      particles_c=particles.copy()
      N=len(S_id) #number of particles handled
      
      # 1D Random Walk function with additional correction term for
      # non-static diffusion after Uffink 1990 p.15 & p.24ff and Kitanidis 1994
      xi=np.random.rand(N,2)*2.-1. #this gives a uniform distribution

      cells=particles.cell.loc[S_id].astype(np.intp).values
      s_cells=mc.soilgrid.ravel()[cells]-1
      
      theta=mc.soilmatrix.ts[s_cells].values*thS.ravel()[cells]/100.
      #psi_euler=mc.psi100[thS.ravel()[cells],s_cells]
      particles.loc[S_id,'LTEbin']=np.fmin(particles.LTEbin.loc[S_id].astype(np.intp).values,len(mc.ku)-1) #debug LTE too large
      u=mc.ku[particles.LTEbin.loc[S_id].astype(np.intp).values,s_cells]*theta
      D=mc.D[particles.LTEbin.loc[S_id].astype(np.intp).values,s_cells]*theta


      #debug:
      #if len(np.shape(dev_uni))>1:
      #  print(deviation)
      #  print(dev_uni)

      #try: 
      #  all(dev_uni>=0.)
      #  #good
      #except:
      #  print(deviation)
      #  print(dev_uni)

      #if not all(dev_uni>=0.):
      #  dev_uni[~(dev_uni>=0.)]=1.
      #  #all fine
      #  print('all fine')
      #else:
      #  print(dev_uni.astype(str))
      #  dev_uni[~(dev_uni>=0.)]=1.
      #  print('-------------------------------')
      #  print(dev_uni.astype(str))
      #xi*=np.reshape(np.repeat(dev_uni[cells],2).T,(N,2))

      #if not all(xi.ravel()>=-1.):
      #  print('PROBLEM: xi turned nan')
      
      #project step with dt based on starting conditions
      vert_sproj=(mc.dt*u + (xi[:,0]*((6*D*mc.dt)**0.5)))
      lat_sproj=(xi[:,1]*((6*D*mc.dt)**0.5))

      #Itô Scheme after Uffink 1990 and Kitanidis 1994 for vertical step
      #modified Ito Scheme after Kitanidis 1994 for lateral step
      dx=np.sqrt(vert_sproj**2+lat_sproj**2)
      
      # project step and updated state
      # new positions
      particles_c.loc[S_id,'lat'] = particles_c.loc[S_id,'lat'].values + lat_sproj
      particles_c.loc[S_id,'z'] = particles_c.loc[S_id,'z'].values - vert_sproj
      [particles_c.lat,particles_c.z,nodrain]=boundcheck(particles_c.lat.values,particles_c.z.values,mc)
      #here the bins need to update now, since this is the reference to the diffusivity now
      particles_c.cell=cellgrid(particles_c.lat.values,particles_c.z.values,mc)
      
      #particles_c=particles_c.groupby('cell').apply(binupdate_cx,mc)
      particles_c=binupdate_cx2(particles_c, mc)

      [thS_c,npart_c] = gridupdate_thS(particles_c.loc[particles_c.flag==0,'lat'].values,particles_c.loc[particles_c.flag==0,'z'].values,mc)
      if any(particles_c.cell<0):
            print('PROBLEM in cell definition')
            print(particles_c.loc[particles_c.cell<0])
      if any(np.isnan(particles_c.cell)):
            print('NaN in cell definition')
            print(particles_c.lat[np.isnan(particles_c.cell)])
            print(particles_c.z[np.isnan(particles_c.cell)])
            print(particles_c.cell[np.isnan(particles_c.cell)])
      s_cells=mc.soilgrid.ravel()[particles_c.cell.loc[S_id].astype(np.intp).values]-1
      theta=mc.soilmatrix.ts[s_cells].values*thS_c.ravel()[particles_c.cell.loc[S_id].astype(np.intp).values]/100.
      #psi_euler2=mc.psi100[thS_c.ravel()[particles_c.cell.loc[S_id].astype(np.intp).values],s_cells]

      u_proj=mc.ku[particles_c.LTEbin.loc[S_id].astype(np.intp).values,s_cells]*theta
      D_proj=mc.D[particles_c.LTEbin.loc[S_id].astype(np.intp).values,s_cells]*theta

      #psi_gradient=(psi_euler2-psi_euler)/dx
      #psi_gradient[dx==0.]=0.

      #counteract rising diffuivity with rising theta with reduction of available slots
      p_counteract=np.ones(N)
      openslots_grid = (mc.mxbin-npart_c).astype(np.float) #open slots in target cell
      openslots_grid = np.fmin(openslots_grid/(mc.mxbin-mc.FC[mc.soilgrid-1]).astype(np.float),1.)
      LTEref = particles.loc[S_id].LTEbin-particles_c.loc[S_id].LTEbin
      idx = (LTEref<0).values #LTEbin increase
      p_counteract[idx]=openslots_grid.ravel()[particles_c.loc[S_id].loc[LTEref<0].cell]
      p_counteract**mc.counteract_pow

      #Stratonovich:
      #corrD=np.abs(D_proj-D)/dx
      #corrD[dx==0.]=0.
      D_mean=np.sqrt(D_proj*D)
      u_mean=np.sqrt(u_proj*u)
      #corru=np.sqrt(u_proj*u)
      ##corrD[corrD>corru]=corru[corrD>corru]
      # corrected step
      #vert_sproj=((corru-corrD)*mc.dt + (xi[:,0]*((6*D_mean*mc.dt)**0.5)))
      #lat_sproj=(xi[:,1]/np.abs(xi[:,1]))*corrD*mc.dt + (xi[:,1]*((6*D_mean*mc.dt)**0.5))

      #Ito:
      #corrD=np.abs(D_proj-D)/dx
      #corrD[dx==0.]=0.

      # add a pressure gradient dependency
      #vert_sproj*=psi_gradient
      #lat_sproj*=psi_gradient

      #LTE version does not require correction
      #corrD*=0.

      #vert_sproj=((u-corrD)*mc.dt + (xi[:,0]*((6*D*mc.dt)**0.5)))
      #lat_sproj=(xi[:,1]/np.abs(xi[:,1]))*corrD*mc.dt + (xi[:,1]*((6*D_mean*mc.dt)**0.5))
      vert_sproj=((u_mean)*mc.dt + (p_counteract*xi[:,0]*((6*D_mean*mc.dt)**0.5)))
      lat_sproj=(p_counteract*xi[:,1]*((6*D_mean*mc.dt)**0.5))

      # saturation check
      # if this cases oversaturation at the target cell > balance incoming outgoing pressure


      # new positions
      particles.loc[S_id,'lat'] = particles.loc[S_id,'lat'].values + lat_sproj
      particles.loc[S_id,'z'] = particles.loc[S_id,'z'].values - vert_sproj
      [particles.lat,particles.z,nodrain]=boundcheck(particles.lat.values,particles.z.values,mc)
      particles.cell=cellgrid(particles.lat.values,particles.z.values,mc)

      if any(~nodrain):
          particles.flag.iloc[~nodrain]=len(mc.maccols)+1

      return







