# coding=utf-8

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import scipy.constants as const
import vG_conv as vG
import partdyn_d2 as pdyn
# http://docs.scipy.org/doc/scipy/reference/constants.html#module-scipy.constants


# MODEL INI & DATA READ
# import mcini as mc

def cdf(x):
    scol = np.sum(x)
    if scol>0:
        result = x/scol
    else:
        result = x*0
    return result

def minloc(x):
    result=np.min(np.where(x==0))
    return result


def waterdensity(T,P):
    '''
       Calc density of water depending on T [C] and P [Pa]
       defined between 0 and 40 C and given in g/m3
       Thiesen Equation after CIPM
       Tanaka et al. 2001, http://iopscience.iop.org/0026-1394/38/4/3

       NOTE: the effect of solved salts, isotopic composition, etc. remain
       disregarded here. especially the former will need to be closely
       considerd in a revised version! DEBUG.
       
       INPUT:  Temperature T in C as numpy.array
               Pressure P in Pa as numpy.array (-9999 for not considered)
       OUTPUT: Water Density in g/m3
       
       EXAMPLE: waterdensity(np.array((20,21,42)),np.array(-9999.))
       (cc) jackisch@kit.edu
    '''
    
    import numpy as np
    
    # T needs to be given in C
    a1 = -3.983035  # C        
    a2 = 301.797    # C        
    a3 = 522528.9   # C2       
    a4 = 69.34881   # C        
    a5 = 999974.950 # g/m3
    
    dens=a5*(1-((T+a1)**2*(T+a2))/(a3*(T+a4)))
    
    # P needs to be given in Pa
    # use P=-9999 if pressure correction is void
    if P.min()>-9999:
        c1 = 5.074e-10   # Pa-1     
        c2 = -3.26e-12   # Pa-1 * C-1  
        c3 = 4.16e-15    # Pa-1 * C-2  .
        Cp = 1 + (c1 + c2*T + c3*T**2) * (P - 101325)
        dens=dens*Cp
    
    # remove values outside definition bounds and set to one
    if (((T.min()<0) | (T.max()>40)) & (T.size>1)):
        idx=np.where((T<0) | (T>40))
        dens[idx]=100000.0  #dummy outside defined bounds 
    
    return dens

def mc_diffs(mc,bins=101,psibins=121):
    '''Calculate diffs for D calculation
    '''
    #bin based:
    dpsidthetamx=np.empty((bins,mc.soilmatrix.no.size))
    D=np.empty((bins,mc.soilmatrix.no.size))
    Dcalc=np.empty(bins)
    ku=np.empty(bins)
    kumx=np.empty((bins,mc.soilmatrix.no.size))
    psi=np.empty(bins)
    psimx=np.empty((bins,mc.soilmatrix.no.size))
    theta=np.empty(bins)
    thetamx=np.empty((bins,mc.soilmatrix.no.size))
    dpsidtheta=np.empty(bins)
    cH2O=np.empty(bins)
    cH2Omx=np.empty((bins,mc.soilmatrix.no.size))
    

    for i in np.arange(mc.soilmatrix.no.size):
        for j in np.arange(bins):
            thetaS=float(j)/(bins-1)
            #ku[j]=mc.soilmatrix.ks[i]*thetaS**0.5*(1.-(1.-thetaS**(1./mc.soilmatrix.m[i]))**mc.soilmatrix.m[i])**2.
            psi[j]=vG.psi_thst(thetaS,mc.soilmatrix.alpha[i],mc.soilmatrix.n[i],mc.soilmatrix.m[i])
            ku[j]=vG.ku_psi(psi[j], mc.soilmatrix.ks[i], mc.soilmatrix.alpha[i], mc.soilmatrix.n[i], mc.soilmatrix.m[i])
            
            #psi=psi/100. #convert to [m]
            #theta[j]=thetaS*(mc.soilmatrix.ts[i]-mc.soilmatrix.tr[i])+mc.soilmatrix.tr[i]
            theta[j]=vG.theta_thst(thetaS,mc.soilmatrix.ts[i],mc.soilmatrix.tr[i])
            dpsidtheta[j]=vG.dpsidtheta_thst(thetaS,mc.soilmatrix.ts[i],mc.soilmatrix.tr[i],mc.soilmatrix.alpha[i],mc.soilmatrix.n[i], mc.soilmatrix.m[i])
            cH2O[j]=vG.c_psi(psi[j],mc.soilmatrix.ts[i],mc.soilmatrix.tr[i],mc.soilmatrix.alpha[i], mc.soilmatrix.n[i], mc.soilmatrix.m[i])
            #dummy=-mc.soilmatrix.m[i]*(1./(1.+np.abs((psi[j])*mc.soilmatrix.alpha[i])**mc.soilmatrix.n[i]))**(mc.soilmatrix.m[i]+1) *mc.soilmatrix.n[i]*(abs(psi[j])*mc.soilmatrix.alpha[i])**(mc.soilmatrix.n[i]-1.)*mc.soilmatrix.alpha[i]
            #cH2O[j]=-(mc.soilmatrix.ts[i]-mc.soilmatrix.tr[i])*dummy
            Dcalc[j]=vG.D_psi(psi[j],mc.soilmatrix.ks[i],mc.soilmatrix.ts[i],mc.soilmatrix.tr[i],mc.soilmatrix.alpha[i], mc.soilmatrix.n[i], mc.soilmatrix.m[i])
        psi[0]=-1.0e+11
        theta[theta<0.01]=0.01
        #DI=(ku[:-1]+(np.diff(ku)/2.))*np.diff(psi)/np.diff(theta)
        #DI[0]=0 #define very low diffusion at zero
        D[:,i]=Dcalc#/10000. # convert cm2/s -> m2/s
        D[-1,:]=D[-2,:]
        psimx[:,i]=psi
        thetamx[:,i]=theta
        dpsidthetamx[:,i]=dpsidtheta
        kumx[:,i]=ku
        cH2Omx[:,i]=cH2O

    #DEBUG: avoid nan in definitions. set to minimum
    D[np.isnan(D)] = np.amin(D[np.reshape(np.isnan(D),np.shape(D))])
    [ax,ay]=np.where(D>1)
    #D[ax,ay]=np.sqrt(D[ax-1,ay]*D[ax+1,ay])
    axp1=np.fmin(bins-1,ax+1)
    axm1=np.fmax(0,ax-1)
    D[ax,ay]=10**(0.5*(np.log10(D[axm1,ay])+np.log10(D[axp1,ay])))
    psimx[np.isnan(psimx)] = np.amin(psimx[np.reshape(~np.isnan(psimx),np.shape(psimx))])
    kumx[np.isnan(kumx)] = np.amin(kumx[np.reshape(~np.isnan(kumx),np.shape(kumx))])
    cH2Omx[np.isnan(cH2Omx)] = np.amin(cH2Omx[np.reshape(~np.isnan(cH2Omx),np.shape(cH2Omx))])
    dpsidthetamx[np.isnan(dpsidthetamx)] = np.amin(dpsidthetamx[np.reshape(~np.isnan(dpsidthetamx),np.shape(dpsidthetamx))])

    mc.D=np.abs(D)
    mc.psi=psimx
    mc.theta=thetamx
    mc.ku=kumx
    mc.cH2O=cH2Omx
    mc.dpsidtheta=dpsidthetamx

    #thetaS based:
    bins=101
    dpsidthetamx=np.empty((bins,mc.soilmatrix.no.size))
    D=np.empty((bins,mc.soilmatrix.no.size))
    Dcalc=np.empty(bins)
    ku=np.empty(bins)
    kumx=np.empty((bins,mc.soilmatrix.no.size))
    psi=np.empty(bins)
    psimx=np.empty((bins,mc.soilmatrix.no.size))
    theta=np.empty(bins)
    thetamx=np.empty((bins,mc.soilmatrix.no.size))
    dpsidtheta=np.empty(bins)
    cH2O=np.empty(bins)
    cH2Omx=np.empty((bins,mc.soilmatrix.no.size))
    

    for i in np.arange(mc.soilmatrix.no.size):
        for j in np.arange(bins):
            thetaS=float(j)/(bins-1)
            #ku[j]=mc.soilmatrix.ks[i]*thetaS**0.5*(1.-(1.-thetaS**(1./mc.soilmatrix.m[i]))**mc.soilmatrix.m[i])**2.
            psi[j]=vG.psi_thst(thetaS,mc.soilmatrix.alpha[i],mc.soilmatrix.n[i],mc.soilmatrix.m[i])
            ku[j]=vG.ku_psi(psi[j], mc.soilmatrix.ks[i], mc.soilmatrix.alpha[i], mc.soilmatrix.n[i], mc.soilmatrix.m[i])
            #dpsi=vG.psi_thst(np.amin([0.0001,thetaS-0.001]),mc.soilmatrix.alpha[i],mc.soilmatrix.n[i],mc.soilmatrix.m[i])-vG.psi_thst(np.amax([0.9999,thetaS+0.001]),mc.soilmatrix.alpha[i],mc.soilmatrix.n[i],mc.soilmatrix.m[i])
            
            #psi=psi/100. #convert to [m]
            #theta[j]=thetaS*(mc.soilmatrix.ts[i]-mc.soilmatrix.tr[i])+mc.soilmatrix.tr[i]
            theta[j]=vG.theta_thst(thetaS,mc.soilmatrix.ts[i],mc.soilmatrix.tr[i])
            #dtheta=vG.theta_thst(np.amin([0.0001,thetaS-0.001]),mc.soilmatrix.ts[i],mc.soilmatrix.tr[i])-vG.theta_thst(np.amax([0.9999,thetaS+0.001]),mc.soilmatrix.ts[i],mc.soilmatrix.tr[i])
            #dpsidtheta[j]=dpsi/dtheta
            dpsidtheta[j]=vG.dpsidtheta_thst(thetaS,mc.soilmatrix.ts[i],mc.soilmatrix.tr[i],mc.soilmatrix.alpha[i],mc.soilmatrix.n[i], mc.soilmatrix.m[i])
            cH2O[j]=vG.c_psi(psi[j],mc.soilmatrix.ts[i],mc.soilmatrix.tr[i],mc.soilmatrix.alpha[i], mc.soilmatrix.n[i], mc.soilmatrix.m[i])
            #dummy=-mc.soilmatrix.m[i]*(1./(1.+np.abs((psi[j])*mc.soilmatrix.alpha[i])**mc.soilmatrix.n[i]))**(mc.soilmatrix.m[i]+1) *mc.soilmatrix.n[i]*(abs(psi[j])*mc.soilmatrix.alpha[i])**(mc.soilmatrix.n[i]-1.)*mc.soilmatrix.alpha[i]
            #cH2O[j]=-(mc.soilmatrix.ts[i]-mc.soilmatrix.tr[i])*dummy
            Dcalc[j]=vG.D_psi(psi[j],mc.soilmatrix.ks[i],mc.soilmatrix.ts[i],mc.soilmatrix.tr[i],mc.soilmatrix.alpha[i], mc.soilmatrix.n[i], mc.soilmatrix.m[i])
        psi[0]=-1.0e+11
        theta[theta<0.01]=0.01
        #DI=(ku[:-1]+(np.diff(ku)/2.))*np.diff(psi)/np.diff(theta)
        #DI[0]=0 #define very low diffusion at zero
        D[:,i]=Dcalc#/10000. # convert cm2/s -> m2/s
        D[-1,:]=D[-2,:]
        psimx[:,i]=psi
        thetamx[:,i]=theta
        dpsidthetamx[:,i]=dpsidtheta
        kumx[:,i]=ku
        cH2Omx[:,i]=cH2O

#DEBUG: avoid nan in definitions. set to minimum
    D[np.isnan(D)] = np.amin(D[np.reshape(~np.isnan(D),np.shape(D))])
    psimx[np.isnan(psimx)] = np.amin(psimx[np.reshape(~np.isnan(psimx),np.shape(psimx))])
    kumx[np.isnan(kumx)] = np.amin(kumx[np.reshape(~np.isnan(kumx),np.shape(kumx))])
    cH2Omx[np.isnan(cH2Omx)] = np.amin(cH2Omx[np.reshape(~np.isnan(cH2Omx),np.shape(cH2Omx))])
    dpsidthetamx[np.isnan(dpsidthetamx)] = np.amin(dpsidthetamx[np.reshape(~np.isnan(dpsidthetamx),np.shape(dpsidthetamx))])

    mc.D100=np.abs(D)
    mc.psi100=psimx
    mc.theta100=thetamx
    mc.ku100=kumx
    mc.cH2O100=cH2Omx
    mc.dpsidtheta100=dpsidthetamx


#       However, we may need a psi based approach since this 
#       is establishing the respective gradient.
    ku=np.empty(psibins)
    p_kumx=np.empty((psibins,mc.soilmatrix.no.size))
    theta=np.empty(psibins)
    p_thetamx=np.empty((psibins,mc.soilmatrix.no.size))
    cH2O=np.empty(psibins)
    p_cH2Omx=np.empty((psibins,mc.soilmatrix.no.size))

    for i in np.arange(mc.soilmatrix.no.size):
        for j in np.arange(psibins):
            psi=-10**((float(j)/(psibins/10.))-2.)
            v = 1. + (mc.soilmatrix.alpha[i]* np.abs(psi))**mc.soilmatrix.n[i]
            ku[j] = (mc.soilmatrix.ks[i] * (1. - ((mc.soilmatrix.alpha[i]*np.abs(psi))**(mc.soilmatrix.n[i]-1))*(v**(-mc.soilmatrix.m[i])) )**2. / (v**(mc.soilmatrix.m[i]*0.5)))/3600.
            thetaS = (1./(1.+(psi*mc.soilmatrix.alpha[i])**mc.soilmatrix.n[i]))**mc.soilmatrix.m[i]
            theta[j]=thetaS*(mc.soilmatrix.ts[i]-mc.soilmatrix.tr[i])+mc.soilmatrix.tr[i]
            dummy=-mc.soilmatrix.m[i]*(1./(1.+np.abs(psi*mc.soilmatrix.alpha[i])**mc.soilmatrix.n[i]))**(mc.soilmatrix.m[i]+1.) *mc.soilmatrix.n[i]*(np.abs(psi)*mc.soilmatrix.alpha[i])**(mc.soilmatrix.n[i]-1.)*mc.soilmatrix.alpha[i]
            cH2O[j]=-(mc.soilmatrix.ts[i]-mc.soilmatrix.tr[i])*dummy

        p_thetamx[:,i]=theta
        p_kumx[:,i]=ku
        p_cH2Omx[:,i]=cH2O

    mc.p_th=p_thetamx
    mc.p_ku=p_kumx
    mc.p_cH2O=p_cH2Omx

    #get FC at psi = -0.33 bar
    #pressure conversion: m_H2O = psi_bar * (100000. Pa/bar / 9806.65 Pa/m)
    dummy=np.zeros(len(mc.psi[0,:]))
    for i in range(len(dummy)):
        dummy[i]=np.argmin(np.abs(-mc.psi[:,i]-(0.33*(100./9.80665))))
    mc.FC=dummy.astype(int)
    
    #debug: for step-like theoretic retention curves FC can get zero which is troublesome
    mc.FC[mc.FC<=mc.part_sizefac/25.]=int(mc.part_sizefac/25.)

    return mc

def ini_bins(mc):
    #introduce bins 
    #DEBUG: this may make some of the former routines obsolete. CHECK.
    #get max bin as share of theta(sat)
    mc.mxbin=np.round(mc.soilmatrix.ts.values[mc.soilgrid-1]*np.abs(mc.gridcellA/mc.particleA).values[0]).astype(np.intp)
    return mc


def dataread_caos(mc):
    macbase=pd.read_csv(mc.macbf, sep=' ')
    tracerbase=pd.read_csv(mc.tracerbf, sep='\t')
    soilmatrix=pd.read_csv(mc.matrixbf, sep=' ')
    mc.soilmatrix=soilmatrix

    #calculate missing van genuchten m
    #if -any(mc.soilmatrix.columns=='m'):
    mc.soilmatrix['m'] = np.fmax(1-1/mc.soilmatrix.n,0.1)
    
    #covert tracer profile into advective velocity distribution
    #tracer concentration is used as proxy :: columnwise normalisation
    tracerbase=pd.read_csv(mc.tracerbf, sep='\t')

    t_cdf=tracerbase.apply(cdf,axis=0)
    #this is the cdf of normalised tracer concentrations

    #FIGURE
    #fig, ax = plt.subplots()
    #heatmap = ax.pcolor(t_cdf, cmap=pylab.cm.Blues, alpha=0.8)
    #ax.invert_yaxis()

    mc.a_velocity=np.arange(-mc.tracer_vertgrid/2,-mc.tracer_vertgrid*(len(tracerbase)+0.5),-mc.tracer_vertgrid)/mc.tracer_t_ex
    #this is the vector of velocities

    ## DEVISION INTO FAST AND SLOW HUMP
    cutrow=np.min(t_cdf.apply(minloc,axis=0))
    #debug!!!
    cutrow=0

    #slow hump
    mc.t_cdf_slow=t_cdf[0:cutrow].apply(cdf,axis=0)

    #fast hump
    mc.t_cdf_fast=t_cdf[cutrow+1:].apply(cdf,axis=0)

    #GET MACROPORE INI FUNCTIONS
    import macropore_ini as mpo

    if mc.nomac==True:
        mc.macshare=pd.Series(0.0)
        mac=pd.DataFrame([dict(no=1, share=0.00001, 
                                 minA=0., maxA=0., meanA=0., medianA=0.,
                                 minP=0., maxP=0., meanP=0., medianP=0.,
                                 minDia=0., maxDia=0., meanDia=0., medianDia=0.,
                                 minmnD=1., maxmnD=1., meanmnD=1., medianmnD=1.,
                                 minmxD=1., maxmxD=1., meanmxD=1., medianmxD=1.,
                                 depth=mc.soildepth), ])  
        mc.a_velocity_real=0. #no macs, no mac_velocity.
        mc=mpo.mac_matrix_setup2(mac,mc)
        mc.mac=mac
    
    elif mc.nomac=='Single':
        mc.macshare=pd.Series(0.001).repeat(len(mc.a_velocity))
        mac=pd.DataFrame([dict(no=1, share=0.001, 
                                 minA=0.0001, maxA=0.0001, meanA=0.0001, medianA=0.0001,
                                 minP=0.01*np.pi, maxP=0.01*np.pi, meanP=0.01*np.pi, medianP=0.01*np.pi,
                                 minDia=0.01, maxDia=0.01, meanDia=0.01, medianDia=0.01,
                                 minmnD=0.5, maxmnD=0.5, meanmnD=0.5, medianmnD=0.5,
                                 minmxD=0.5, maxmxD=0.5, meanmxD=0.5, medianmxD=0.5,
                                 depth=mc.soildepth), ])  
        #since we derive macropore flow velocity from tracer breakthrough, this is already the real velocity!
        mc.a_velocity_real=mc.a_velocity
        mc=mpo.mac_matrix_setup2(mac,mc)
        mc.mac=mac
    
    elif mc.nomac=='Image':
        #READ MACROPORE DATA FROM IMAGE FILES
        mac=pd.read_csv(mc.macimg, sep=',')
        patch_def=mpo.macfind_g(mac.file[0],[mac.threshold_l[0],mac.threshold_u[0]])
        patch_dummy=patch_def.copy()*0.

        for i in np.arange(len(mac)-1)+1:
            patchnow=mpo.macfind_g(mac.file[i],[mac.threshold_l[i],mac.threshold_u[i]])
            if isinstance(patchnow,pd.DataFrame):
                patch_def=patch_def.append(patchnow)
            else:
                patch_def=patch_def.append(patch_dummy)

        #join macropore definitions
        patch_def=patch_def.set_index(np.arange(len(mac)))
        mac=mac.join(patch_def)
        mc.mac=mac
        mc=mpo.mac_matrix_setup(mac,mc)

    else:
        #READ MACROPORES STATISTICS
        mc=mpo.mac_matrix_setup2(mc.macdef,mc)

    #precautional measure:
    mc.mgrid.vertgrid=int(mc.mgrid.vertgrid)
    mc.mgrid.latgrid=int(mc.mgrid.latgrid)

    #get macropore share at advection vector
    z_centroids=np.arange(-mc.tracer_vertgrid/2,-mc.tracer_vertgrid*(len(tracerbase)+0.5),-mc.tracer_vertgrid)
    share_idx=(z_centroids*0.).astype(int)
    for zc in np.arange(len(z_centroids)):
        if -z_centroids[zc]>=mc.md_depth[0]:
            share_idx[zc]=np.where(mc.md_depth<=-z_centroids[zc])[0][-1]+1
            if -z_centroids[zc]>mc.md_depth[-2]:
                share_idx[zc]=len(mc.md_share)-1
        else:
            share_idx[zc]=0
    
    share_idx[share_idx>=len(mc.md_share)]=len(mc.md_share)-1
    
    if mc.nomac==True:
        mc.macshare=np.nan
        mc.a_velocity_real=np.array([0.])
    else:
        mc.macshare=mc.md_share[share_idx]
    
        #calculate cumulative velocity in macropores only (cumulate areal share of macropores at resprective depth)
        a_velocity_real=-np.cumsum(np.cumsum(mc.tracer_vertgrid/mc.macshare)/((np.arange(len(mc.macshare))+1)*mc.tracer_t_ex))
        mc.a_velocity_real=a_velocity_real

    if (mc.nomac==False) | (mc.nomac=='Image'):
        mpo.mac_plot(mc.macP,mc)
    mc=mc_diffs(mc)
    mc.prects=False

    mc.mac_shape=np.zeros((mc.mgrid.vertgrid.values[0],len(mc.maccols)))
    mc.mac_volume=np.zeros((mc.mgrid.vertgrid.values[0],len(mc.maccols)))
    mc.mac_contact=np.zeros((mc.mgrid.vertgrid.values[0],len(mc.maccols)))
    for i in np.arange(mc.mgrid.vertgrid.values[0]):
        idx=np.where(mc.md_depth>-mc.mgrid.vertfac.values*i)[0]
        if len(idx)==0:
            idx=np.shape(mc.md_area)[1]-1
        else:
            idx=int(np.amin([idx[0],np.shape(mc.md_area)[1]-1]))

        mc.mac_shape[i,:]=mc.md_pshape[:,idx]
        mc.mac_volume[i,:]=mc.mgrid.vertgrid.values*mc.md_area[:,idx]/mc.refarea
        mc.mac_contact[i,:]=mc.md_pshape[:,idx]*np.sqrt(mc.md_area[:,idx])

    mc.gridcellA=mc.mgrid.vertfac*mc.mgrid.latfac
    mc.maccap=np.ceil((2.*mc.md_area/(-mc.gridcellA.values*mc.mgrid.latgrid.values))*mc.part_sizefac)
    mc.mactopfill=np.ceil((2.*mc.md_area/(-mc.gridcellA.values*mc.mgrid.latgrid.values))*mc.part_sizefac)[:,0]*0. #all empty
    try:
        mc.macposix = np.unique(mc.macP[0].exterior.coords.xy[1])[::-1]  #reference position (z) of macropore capacity 
    except:
        mc.macposix = []

    print('MODEL SETUP READY.')
    return mc #[mc,soilmatrix,macP,mac,macid,macconnect,soilgrid,matrixdef,mc.mgrid]

def passign_initals(j,k,rw,cl,mcl,mcv):
    x=(cl+np.random.rand(j))*mcl
    z=(rw+np.random.rand(j))*mcv
    LTEbin=np.arange(j)
    return pd.DataFrame([z,x,np.repeat(k,j),LTEbin]).T

def particle_setup(mc,paral=False,newpart=True):
    # define particle size
    # WARNING: as in any model so far, we have a volume problem here. 
    #          we consider all parts of the domain as static in volume at this stage. 
    #          however, we will work on a revision of this soon.
    mc.gridcellA=mc.mgrid.vertfac*mc.mgrid.latfac
    mc.particleA=abs(mc.gridcellA.values)/(2*mc.part_sizefac) #assume average ks at about 0.5 as reference of particle size
    mc.particleD=2.*np.sqrt(mc.particleA/np.pi)
    mc.particleV=3./4.*np.pi*(mc.particleD/2.)**3.
    mc.particleD/=np.sqrt(abs(mc.gridcellA.values))
    mc.particleV/=np.sqrt(abs(mc.gridcellA.values)) #assume grid size as 3rd dimension
    mc.particlemass=waterdensity(np.array(20),np.array(-9999))*mc.particleV #assume 20C as reference for particle mass
                                                                            #DEBUG: a) we assume 2D=3D; b) change 20C to annual mean T?
    #initialise bins and slopes
    mc=ini_bins(mc)
    mc=mc_diffs(mc,np.max(np.max(mc.mxbin)))

    # read ini moist
    inimoistbase=pd.read_csv(mc.inimf, sep=',')
    if any(np.isnan(inimoistbase.theta)):
        #only psi is given - claculate theta
        idx=np.where(np.isnan(inimoistbase.theta).T.values)[0]
        for i in idx:
            inimoistbase.loc[i,'theta']=mc.theta100[np.where(mc.psi100[:,int(np.median(mc.soilgrid))]<=inimoistbase.loc[i,'psi'])[0][-1],int(np.median(mc.soilgrid))]

    if any(np.isnan(inimoistbase.psi)):
        #only psi is given - claculate theta
        idx=np.where(np.isnan(inimoistbase.psi).T.values)[0]
        for i in idx:
            inimoistbase.loc[i,'psi']=mc.psi100[np.where(mc.theta100[:,np.amin([len(mc.soilmatrix)-1,int(np.median(mc.soilgrid))])]<=inimoistbase.loc[i,'theta'])[0][-1],np.amin([len(mc.soilmatrix)-1,int(np.median(mc.soilgrid))])]

    #get bin counts for ini states
    c_soils=np.unique(mc.soilgrid-1)
    ini_idx=pd.DataFrame(np.zeros((len(c_soils),len(inimoistbase))),index=c_soils,columns=inimoistbase.psi.values,dtype=int)
    for i_p in np.arange(len(inimoistbase)):
        for i_s in np.arange(len(c_soils)):
            ini_idx.iloc[i_s,i_p]=np.where(mc.psi[:,c_soils[i_s]]<inimoistbase.psi[i_p])[0][-1]

    #check for thr and ths definitions
    bin_ts=np.round(mc.soilmatrix.ts[c_soils]*np.abs(mc.gridcellA/mc.particleA).values[0]).astype(np.intp)
    bin_tr=np.round(mc.soilmatrix.tr[c_soils]*np.abs(mc.gridcellA/mc.particleA).values[0]).astype(np.intp)

    for i in ini_idx.index:
        ini_idx.loc[i]=np.fmax(np.fmin(ini_idx.loc[i],bin_ts[i]),bin_tr[i])

    #assign npart to correct cells and depths
    s_dummy=(mc.soilgrid.ravel()-1).copy()
    s_dummyx=s_dummy*np.nan
    j=0
    for i in c_soils:
        s_dummyx[s_dummy==i]=j
        j+=1
    
    npart=np.reshape(ini_idx.iloc[s_dummyx.astype(int),0].values,np.shape(mc.soilgrid))
    for i in np.arange(len(inimoistbase))[1:]:
        npart_dummy=np.reshape(ini_idx.iloc[s_dummyx.astype(int),i].values,np.shape(mc.soilgrid))
        npart[mc.zgrid<inimoistbase.loc[i,'zmin']]=npart_dummy[mc.zgrid<inimoistbase.loc[i,'zmin']]
    
    # define macropore capacity based on particle size
    # we introduce a scale factor for converting macropore space and particle size
    # mc.maccap=np.round(mc.md_area/((mc.particleD**2)*np.pi*mc.macscalefac)).astype(int)
    # deprecated!
    
    # estimate macropore capacity as relative share of space (as particle volume is not well-defined for the 1D-2D-3D references)
    mc.maccap=np.ceil((2.*mc.md_area/(-mc.gridcellA.values*mc.mgrid.latgrid.values))*mc.part_sizefac)
    mc.mactopfill=np.ceil((2.*mc.md_area/(-mc.gridcellA.values*mc.mgrid.latgrid.values))*mc.part_sizefac)[:,0]*0. #all empty
    # assumption: the pore space is converted into particles through mc.part_sizefac. this is now reprojected to the macropore by using the areal share of the macropores
    # DEBUG: there is still some inconcistency about the areas and volumes, but it may be a valid estimate with rather few assumptions

    
    # setup particle domain
    particles=pd.DataFrame(np.zeros(int(np.sum(npart))*8).reshape(int(np.sum(npart)),8),columns=['lat', 'z', 'conc', 'temp', 'age', 'flag', 'fastlane', 'advect'])
    particles['cell']=pd.Series(np.zeros(int(np.sum(npart)),dtype=int), index=particles.index)
    particles['LTEbin']=pd.Series(np.zeros(int(np.sum(npart)),dtype=int), index=particles.index)
    particles['exfilt']=pd.Series(np.zeros(int(np.sum(npart)),dtype=int), index=particles.index)
    # distribute particles
    k=0
    npartr=npart.ravel()
    cells=len(npartr)
    rw,cl=np.unravel_index(np.arange(cells),(int(mc.mgrid.vertgrid),int(mc.mgrid.latgrid)))

    if newpart:
        if paral:
            try:
                from joblib import Parallel, delayed
                import multiprocessing
                num_cores = multiprocessing.cpu_count()
                results = Parallel(n_jobs=num_cores)(
                    delayed(passign_initals)(npartr[i], i, rw[i], cl[i], mc.mgrid.latfac.values,
                                             mc.mgrid.vertfac.values) for i in np.arange(cells))
                results = pd.concat(results)
                particles.iloc[:, [0, 1]] = results[[1, 0]].values
                particles.iloc[:, [8, 9]] = results[[2, 3]].values.astype(np.intp)

            except:
                print('parallel processing python packages failed. initialising particles sequentially.')
                for i in np.arange(cells):
                    j=int(npartr[i])
                    particles.iloc[k:(k+j),8]=i #cell
                    rw,cl=np.unravel_index(i,(mc.mgrid.vertgrid[0],mc.mgrid.latgrid[0]))
                    particles.iloc[k:(k+j),0]=(cl+np.random.rand(j))*mc.mgrid.latfac[0]
                    particles.iloc[k:(k+j),1]=(rw+np.random.rand(j))*mc.mgrid.vertfac[0]
                    particles.iloc[k:(k+j),9]=np.arange(j) #LTEbin
                    k+=j

        else:
            for i in np.arange(cells):
                j=int(npartr[i])
                particles.iloc[k:(k+j),8]=i #cell
                rw,cl=np.unravel_index(i,(mc.mgrid.vertgrid[0],mc.mgrid.latgrid[0]))
                particles.iloc[k:(k+j),0]=(cl+np.random.rand(j))*mc.mgrid.latfac[0]
                particles.iloc[k:(k+j),1]=(rw+np.random.rand(j))*mc.mgrid.vertfac[0]
                particles.iloc[k:(k+j),9]=np.arange(j) #LTEbin
                k+=j

        particles.fastlane=np.random.randint(len(mc.t_cdf_fast.T), size=len(particles))
        particles.advect=pdyn.assignadvect(int(np.sum(npart)),mc,particles.fastlane.values,True)

    mc.mgrid['cells']=cells
    return [mc,particles,npart]


