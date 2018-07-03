import numpy as np
import pandas as pd
import scipy as sp
import scipy.constants as const
import os, sys

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def loadconnect(pathdir='./', mcinif='mcini', oldvers=False, experimental=False):
    lib_path = os.path.abspath(pathdir)
    sys.path.append(lib_path)

    import dataread as dr
    if oldvers:
        import mcpickle as mcp
    else:
        if sys.version_info[0]==3:
            import mcpickle3 as mcp
        else:
            import mcpickle2 as mcp
    import infilt as cinf
    if experimental:
        import partdyn_d8 as pdyn
    else:
        import partdyn_d2 as pdyn
    mc = __import__(mcinif)
    import vG_conv as vG

    return(dr,mc,mcp,pdyn,cinf,vG)

def preproc_echoRD(mc, dr, mcp, pickfile='test.pickle'):
    mc=dr.dataread_caos(mc)
    mcp.mcpick_in(mc,pickfile)
    return mc

def pickup_echoRD(mc, mcp, dr, pickfile='test.pickle'):
    mcp.mcpick_out(mc,pickfile)
    [mc,particles,npart]=dr.particle_setup(mc,False)
    precTS=pd.read_csv(mc.precf, sep=',',skiprows=3)

    return(mc,particles,npart,precTS)

def particle_setup_obs(theta_obs,mc,vG,dr,pdyn):
    moistdomain=np.tile(theta_obs,int(mc.mgrid.latgrid)).reshape((mc.mgrid.latgrid,mc.mgrid.vertgrid)).T
        
    # define particle size
    # WARNING: as in any model so far, we have a volume problem here. 
    #          we consider all parts of the domain as static in volume at this stage. 
    #          however, we will work on a revision of this soon.
    mc.gridcellA=mc.mgrid.vertfac*mc.mgrid.latfac
    mc.particleA=abs(mc.gridcellA.values)/(2*mc.part_sizefac) #assume average ks at about 0.5 as reference of particle size
    mc.particleD=2.*np.sqrt(mc.particleA)/np.pi
    mc.particleV=3./4.*np.pi*(mc.particleD/2.)**3.
    mc.particlemass=dr.waterdensity(np.array(20),np.array(-9999))*mc.particleV #assume 20C as reference for particle mass
                                                                            #DEBUG: a) we assume 2D=3D; b) change 20C to annual mean T?

    # define macropore capacity based on particle size
    # we introduce a scale factor for converting macropore space and particle size
    mc.maccap=np.round(mc.md_area/((mc.particleD**2)*np.pi*mc.macscalefac)).astype(int)

    # convert theta to particles
    # npart=moistdomain*(2*mc.part_sizefac)
    npart=np.floor(mc.part_sizefac*vG.thst_theta(moistdomain,mc.soilmatrix.ts[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid)), mc.soilmatrix.tr[mc.soilgrid.ravel()-1].reshape(np.shape(mc.soilgrid)))).astype(int)

    # setup particle domain
    particles=pd.DataFrame(np.zeros(int(np.sum(npart))*8).reshape(int(np.sum(npart)),8),columns=['lat', 'z', 'conc', 'temp', 'age', 'flag', 'fastlane', 'advect'])
    particles['cell']=pd.Series(np.zeros(int(np.sum(npart)),dtype=int), index=particles.index)
    # distribute particles
    k=0
    npartr=npart.ravel()
    cells=len(npartr)

    for i in np.arange(cells):
        j=int(npartr[i])
        particles.cell[k:(k+j)]=i
        rw,cl=np.unravel_index(i,(mc.mgrid.vertgrid,mc.mgrid.latgrid))
        particles.lat[k:(k+j)]=(cl+np.random.rand(j))*mc.mgrid.latfac.values
        particles.z[k:(k+j)]=(rw+np.random.rand(j))*mc.mgrid.vertfac.values
        k+=j

    particles.fastlane=np.random.randint(len(mc.t_cdf_fast.T), size=len(particles))
    particles.advect=pdyn.assignadvect(int(np.sum(npart)),mc,particles.fastlane.values,True)

    mc.mgrid['cells']=cells
    return [mc,particles.iloc[0:k,:],npart]


def check_lookups(mc):
    if np.isnan(mc.D).any():
        mc.D[np.isnan(mc.D)] = np.nanmin(mc.D)
    if np.isnan(mc.ku).any():
        mc.ku[np.isnan(mc.ku)] = np.nanmin(mc.ku)
    return mc



def CAOSpy_rundx1(tstart,tstop,mc,pdyn,cinf,precTS,particles,leftover=0,drained=0.1,dt_max=1.,splitfac=10,prec_2D=False,maccoat=10.,exfilt_method='Ediss',saveDT=True,vertcalfac=1.,latcalfac=1.,clogswitch=False,infilt_method='MDA',film=True,infiltscale=False,dynamic_pedo=False,counteractpow=3):
    if run_from_ipython():
        from IPython import display

    timenow=tstart
    prec_part=0. #precipitation which is less than one particle to accumulate
    acc_mxinf=0. #matrix infiltration may become very small - this shall handle that some particles accumulate to infiltrate
    exfilt_p=0. #exfiltration from the macropores
    s_red=0.
    mc.splitfac=splitfac
    mc.counteract_pow=counteractpow

    if type(drained)==float:
        drained=pd.DataFrame(np.array([]))
    #loop through time
    while timenow < tstop:
        [thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
        if saveDT==True:
            #define dt as Courant/Neumann criterion
            dt_D=(mc.mgrid.vertfac.values[0])**2 / (6*np.amax(mc.D[particles.LTEbin.max(),:]))
            dt_ku=-mc.mgrid.vertfac.values[0]/np.amax(mc.ku[particles.LTEbin.max(),:])
            dt=np.amin([dt_D,dt_ku,dt_max,tstop-timenow])
            dt=np.amax([dt,0.5])
            mc.dt=dt
        else:
            if type(saveDT)==float:
                #define dt as pre-defined
                dt=np.amin([saveDT,tstop-timenow])
                mc.dt=dt
            elif type(saveDT)==int:
                #define dt as modified  Corant/Neumann criterion
                dt_D=(mc.mgrid.vertfac.values[0])**2 / (6*np.nanmax(mc.D[np.amax(thS),:]))*saveDT
                dt_ku=-mc.mgrid.vertfac.values[0]/np.nanmax(mc.ku[np.amax(thS),:])*saveDT
                dt=np.amin([dt_D,dt_ku,dt_max,tstop-timenow])
                mc.dt=dt
        
        pcount=len(particles)
        #INFILTRATION (new version with binned)
        [p_inf,prec_part,acc_mxinf]=cinf.pmx_infilt(timenow,precTS,prec_part,acc_mxinf,thS,mc,pdyn,mc.dt,leftover,prec_2D,particles.index[-1],infilt_method,infiltscale,npart) #drain all ponding // leftover <-> 0.
        particles=pd.concat([particles,p_inf])
        
        #DEBUG:
        #if len(particles)<100000:
        #    particles.to_pickle('DEBUG_particlesX.pic')
        #    assert False
        #particles.to_pickle('DEBUG_particles.pic')
        #particles=particles.groupby('cell').apply(pdyn.binupdate_pd,mc)
        #particles.loc[particles.LTEbin>(len(mc.ku)-1),'LTEbin']=len(mc.ku)-1
        #plot_mac(particles,'./results/Mtest'+str(np.round(timenow,2)))

        #DIFFUSION
        #[particles,thS,npart,phi_mx]=pdyn.part_diffusion_split(particles,npart,thS,mc,dt,False,splitfac,vertcalfac,latcalfac)
        #[particles,thS,npart,phi_mx]=pdyn.part_diffusion_binned(particles,npart,thS,mc,dt)
        
        [particles,thS,npart,phi_mx]=pdyn.part_diffusion_binned_pd(particles,npart,thS,mc)

        #ADVECTION
        if not particles.loc[(particles.flag>0) & (particles.flag<len(mc.maccols)+1)].empty:
            [particles,s_red,exfilt_p,mc]=pdyn.mac_advectionX(particles,mc,thS,mc.dt,clogswitch,maccoat,exfilt_method,film=film,dynamic_pedo=dynamic_pedo,npart=npart,fc_check=False)

        #INTERACT
        particles=pdyn.mx_mp_interact_nobulk(particles,npart,thS,mc,mc.dt,dynamic_pedo)

        #CLEAN UP DATAFRAME
        drained=drained.append(particles[particles.flag==len(mc.maccols)+1])
        particles=particles.loc[particles.flag!=len(mc.maccols)+1]
        #leftover+=np.intp((particles.z>=0.).sum())
        leftover=particles.z[particles.z>=0.].count()
        particles=particles.loc[particles.z<0.] #drop all leftover back to infilt routine

        if run_from_ipython():
            display.clear_output()
            display.display_pretty(''.join(['time: ',str(timenow),'s  |  n_particles: ',str(pcount),'  |  precip: ',str(len(p_inf)),'  |  advect: ',str(np.sum(((particles.flag>0) & (particles.flag<=len(mc.maccols))))),'  |  exfilt: ',str(int(exfilt_p)),'  |  mean v(adv): ',str(particles.loc[((particles.flag>0) & (particles.flag<=len(mc.maccols))),'advect'].mean())[:7]+str(particles.loc[((particles.flag>0) & (particles.flag<=len(mc.maccols))),'advect'].mean())[-4:],' m/s']))
            display.display_pretty(''.join(['time: ',str(timenow),'s  |  n_particles now: ',str(len(particles)),'  |  leftover: ',str(leftover)]))
        else:
            print('time: ',timenow,'s')

        #particles.loc[particles.cell<0,'cell']=mc.mgrid.cells.values
        #particles=particles[pondparts]
        timenow=timenow+mc.dt

    return(particles,npart,thS,leftover,drained,timenow)

def part_store(particles,mc):
    dummy_n2=pd.Series(np.zeros(mc.mgrid.cells),index=np.arange(mc.mgrid.cells))
    dummy_o2=pd.Series(np.zeros(mc.mgrid.cells),index=np.arange(mc.mgrid.cells))
    dummy_n=particles.loc[((particles.age>0.)),'cell'].value_counts().sort_index()
    dummy_o=particles.loc[((particles.age<=0.)),'cell'].value_counts().sort_index()
    dummy_n2.loc[dummy_n.index]+=dummy_n
    dummy_o2.loc[dummy_o.index]+=dummy_o
    dummy=pd.concat([dummy_n2,dummy_o2],axis=1)
    dummy.columns=['new','old']
    return dummy.values

#define plot routine for macropore states
def plot_mac(particles,savef=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(font='Roboto',style='white', palette='deep',font_scale=1.2)
    zres=50.
    tmacs=int(particles.flag.max())
    for i in particles.flag.unique().astype(np.intp):
        if any(particles.flag==(i+1)):
            plt.subplot(1,tmacs,i+1)
            dummy=np.zeros((np.intp(zres),2))
            dummy[:,0]=np.linspace(0., -1., num=np.intp(zres), endpoint=False)
            filling=np.bincount(np.fmax(0,np.floor(-particles.loc[particles.flag==(i+1),'z']*zres).astype(np.intp)))
            dummy[:len(filling),1]=filling
            plt.plot([0.,0.],[0.,-1.],'grey')
            plt.fill_betweenx(dummy[:,0],0,dummy[:,1],alpha=0.4)
            plt.plot(dummy[:,1],dummy[:,0],'-')
            plt.xlim(0,50)
            plt.ylim(-1.,0.)
            plt.title(str(i))
            plt.xticks([])
            plt.yticks([])
            sns.despine(left=True, bottom=True)
    if savef:
        plt.savefig(savef+'.pdf')

def echoRD_job(mcinif='mcini',mcpick='mc.pickle3',runname='test',
               wdir='./',pathdir='../echoRD/',saveDT=True,
               aref='Shipitalo',LTEdef='instant',infilM='MDA',exfilM='Ediss',
               model_restart=False, parallel=False,hdf5pick=True,update_mf=False,
               update_prec=False,update_part=False,update_stochsoil=False,update_md_area=False,
               update_timenow=False, legacy_pick=False,column=False,
               colref=False,prec2D=False,macscale=1.,
               fsize=(7, 4),w_rat=[4,0.6],h_rat=[1,9]):
    '''
    This is a wrapper for running echoRD easily.
    Be warned that some definitions are implicit in this wrapper

    mcini   -- echoRD ini file of run
    mcpick  -- pickled mode setup
    runname -- name of model run
    wdir    -- working directory path
    pathdir -- path to echoRD model

    Model parameters
    aref    -- Advection reference [Shipitalo | Weiler | Zehe | geogene | geogene2 | obs]
    infilM  -- Infiltration handling [MDA | MED | rand]
    LTEdef  -- Infiltration assumption [instant | ks | random]
    exfilM  -- Exfiltration from macropores [Ediss | RWdiff]
    saveDT  -- optional modified time steps [True | int (factor) | double (static step)]

    model_restart- True: setup new model base and start running, 'only': setup new model base and run one time step, False: reload existing model base, 'joint': reload common base setup
    parallel-- flag for parallel initialisation of particles
    hdf5pick-- flag using hdf5 for pickling

    update_mf- change referenced ini moist file
    update_part- change particle size factor (mc.part_sizefac)
    update_prec- change total precip amount
    update_stochsoil- change stochsoil reference file
    legacy_pick- caompatibility to former runs with different state pickles
    column  -- True if referiing to soil column test cases
    colref  -- reference to circular column instead of 2D pane
    prec2D  -- precipitation definition over area
    macscale-- scale factor for macropore coating (if other than 1)

    fsize   -- plotting parameter canvas size
    w_rat   -- plotting parameter width ratio plot and marginals
    h_rat   -- plotting parameter height ratio marginals and plot
    '''

    import numpy as np
    import pandas as pd
    import scipy as sp
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os, sys
    try:
        import cPickle as pickle
    except:
        import pickle

    try:
        import h5py
    except:
        hdf5pick = False
        print('h5py not available - falling back to pickle use.')
    
    #connect echoRD Tools
    lib_path = os.path.abspath(pathdir)
    sys.path.append(lib_path)
    import vG_conv as vG
    from hydro_tools import plotparticles_t,hydroprofile,plotparticles_gen    #connect to echoRD
    
    import run_echoRD as rE
    #connect and load project
    [dr,mc,mcp,pdyn,cinf,vG]=rE.loadconnect(pathdir='../',mcinif=mcinif,experimental=True)
    if (model_restart==True) | (model_restart=='only'):
        mc=rE.preproc_echoRD(mc, dr, mcp, mcpick)
        import macropore_ini as mpo
        mpo.mac_plot(mc.macP,mc,'macs_'+runname+'.pdf')
    else:
        mc = mcp.mcpick_out(mc,mcpick)

    mc.advectref=aref

    if update_stochsoil:
        mc.stochsoil = update_stochsoil

    try:
        if mc.stochsoil==False:
            hdf5pick = False
            print('h5py not used as mc.stochsoil set False - falling back to pickle use.')
    except:
        mc.stochsoil = False
        hdf5pick = False
        print('h5py not used as mc.stochsoil not given - falling back to pickle use.')

    precTS=pd.read_csv(mc.precf, sep=',',skiprows=3)
    mc.prects=False

    #check for generic versionings of setup:
    if update_prec:
        precTS.total=update_prec
        precTS.intense=precTS.total/(precTS.tend-precTS.tstart)

    if update_part:
        mc.part_sizefac = update_part

    if update_md_area:
        mc.md_area *= update_md_area

    if column:
        total_volume=np.pi*0.5**3
        mc.particleV=total_volume/(mc.mgrid.vertgrid[0]*mc.mgrid.latgrid[0]*(2*mc.part_sizefac))
        mc.prects='column2'
        mc.cloref=colref
        
        if mc.nomac==True:
            #add macropore into soilmatrix for column test
            idx=int(np.shape(mc.soilgrid)[1]/2)
            mc.soilgrid[:,idx-1:idx+1]=13 


    #check for previous runs to hook into
    try:
        if model_restart==True:
            raise ValueError('I will overwrite any existing runfile.')
        #unpickle:
        if hdf5pick:
            particles = pd.read_hdf('./results/P_'+runname+'.h5','table')
            with h5py.File('./results/S_'+runname+'.h5', 'r') as f:
                dset = f['states'][0]
                [t, len_parts, leftover, len_drain, ix] = dset[:]
                if ix>0:
                    ix+=1
                ix = int(ix)
                leftover = int(leftover)
        else:
            with open(''.join([wdir,'/results/Z',runname,'_Mstat.pick']),'rb') as handle:
                pickle_l = pickle.load(handle)
                dummyx = pickle.loads(pickle_l)
                particles = pickle.loads(dummyx[0])
                if legacy_pick:
                    [leftover,drained,t,TSstore,ix] = pickle.loads(dummyx[1])
                    dummy=np.floor(mc.t_end/mc.t_out)
                    mc.mgrid['cells']=np.shape(TSstore)[0]
                    [thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
                    thetastore=np.zeros((int(dummy),mc.mgrid.vertgrid[0],mc.mgrid.latgrid[0]))
                    mc.t_end=24.*3600.
                    mc.t_out=60
                else:
                    [leftover,drained,t,TSstore,thetastore,npart,ix] = pickle.loads(dummyx[1])
                if ix>0:
                    ix+=1
        print('resuming into stored run at t='+str(t)+'...')
        
        if model_restart=='joint':
            #rename runfile now
            runname=runname+str(int(np.round(macscale*10))).zfill(2)

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
        mc.particlemass=dr.waterdensity(np.array(20),np.array(-9999))*mc.particleV #assume 20C as reference for particle mass
                                                                            #DEBUG: a) we assume 2D=3D; b) change 20C to annual mean T?
        #initialise bins and slopes
        mc=dr.ini_bins(mc)
        mc=dr.mc_diffs(mc,np.max(np.max(mc.mxbin)))

        # estimate macropore capacity as relative share of space (as particle volume is not well-defined for the 1D-2D-3D references)
        mc.maccap=np.ceil((2.*mc.md_area/(-mc.gridcellA.values*mc.mgrid.latgrid.values))*mc.part_sizefac)
        mc.mactopfill=np.ceil((2.*mc.md_area/(-mc.gridcellA.values*mc.mgrid.latgrid.values))*mc.part_sizefac)[:,0]*0. #all empty
        # assumption: the pore space is converted into particles through mc.part_sizefac. this is now reprojected to the macropore by using the areal share of the macropores
        # DEBUG: there is still some inconcistency about the areas and volumes, but it may be a valid estimate with rather few assumptions
        mc.mgrid['cells']=len(mc.mxbin.ravel())
        if hdf5pick:
            [thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)

    except:
        if update_mf:
            mc.inimf=update_mf
        #initialise particles
        input()
        [mc,particles,npart]=dr.particle_setup(mc,paral=parallel)
        t=0.
        ix=0
        leftover=0
        try: 
            mc.t_out=mc.t_out
        except:
            mc.t_out=60.
        dummy=np.floor(mc.t_end/mc.t_out)
        TSstore=np.zeros((int(dummy),mc.mgrid.cells[0],2))
        thetastore=np.zeros((int(dummy),mc.mgrid.vertgrid[0],mc.mgrid.latgrid[0]))
        drained=pd.DataFrame(np.array([]))

        if hdf5pick:
            particles.to_hdf('./results/P_'+runname+'.h5','table')
            with h5py.File('./results/S_'+runname+'.h5', 'w') as f:
                dset = f.create_dataset("states", (1,5), dtype='f')
                dset[:] =  [t, len(particles), leftover, len(drained), ix]

        print('starting new run...')

    #check for NaNs in lookup table for first bins and set to next filled ones:
    #DEBUG: actually this should not happen. but it did.
    for i in np.arange(np.shape(mc.D)[1]):
        idx=np.where(np.isnan(mc.D[:,i]))[0]
        mc.D[idx,i]=mc.D[np.amax(idx)+1,i]

    #define bin assignment mode for infiltration particles
    mc.LTEdef=LTEdef #'instant'#'ks' #'instant' #'random'
    mc.LTEmemory=mc.soilgrid.ravel()*0.

    #macropore reference
    mc.maccon=np.where(mc.macconnect.ravel()>0)[0] #index of all connected cells
    mc.md_macdepth=np.abs(mc.md_macdepth)

    #############
    # Run Model #
    #############

    mc.LTEpercentile=70 #new parameter

    t_end=mc.t_end
    output=mc.t_out #mind to set also in TXstore.index definition
    dummy=np.floor(t_end/output)

    infiltmeth=infilM
    exfiltmeth=exfilM
    film=True
    clogswitch=False
    infiltscale=False

    #just store the base files
    if model_restart=='only':
        i=0
        plotparticles_gen(particles,mc,pdyn,vG,runname,t,i,saving=True,relative=False,wdir=wdir,fsize=fsize,w_rat=w_rat,h_rat=h_rat)
        [thS,npart]=pdyn.gridupdate_thS(particles.lat,particles.z,mc)
        
        if hdf5pick:
            with h5py.File(mc.stochsoil, 'r+') as f:
                dset = f["theta"]
                dset[:,:,i]=np.reshape((mc.soilmatrix.loc[mc.soilgrid.ravel()-1,'tr']+(mc.soilmatrix.ts-mc.soilmatrix.tr)[mc.soilgrid.ravel()-1]*thS.ravel()*0.01).values,np.shape(thS))
        
            particles.to_hdf('./results/P_'+runname+'.h5','table')

            with h5py.File('./results/S_'+runname+'.h5', 'r+') as f:
                dset = f["states"]
                dset = [t, len(particles), leftover, len(drained), ix]
            print('wrote initial states to P'+runname+'h5 and S_'+runname+'.h5')
        else:
            TSstore[i,:,:]=rE.part_store(particles,mc)
            thetastore[i,:,:]=np.reshape((mc.soilmatrix.loc[mc.soilgrid.ravel()-1,'tr']+(mc.soilmatrix.ts-mc.soilmatrix.tr)[mc.soilgrid.ravel()-1]*thS.ravel()*0.01).values,np.shape(thS))

            with open(''.join([wdir,'/results/Z',runname,'_Mstat.pick']),'wb') as handle:
               pickle.dump(pickle.dumps([pickle.dumps(particles),pickle.dumps([leftover,drained,t,TSstore,thetastore,npart,i])]), handle, protocol=2)
            print('wrote initial states to Z'+runname+'_Mstat.pick')

    else:
        if update_timenow:
            ix=int(update_timenow/mc.t_out)
            print('Time set manually to '+str(ix*mc.t_out)+'s')
        drained = pd.DataFrame(np.array([])) #new drained dataframe as it is not stored yet
        # loop through plot cycles
        for i in np.arange(dummy.astype(int))[ix:]:
            plotparticles_gen(particles,mc,pdyn,vG,runname,t,i,saving=True,relative=False,wdir=wdir)
            [particles,npart,thS,leftover,drained,t]=rE.CAOSpy_rundx1(i*output,(i+1)*output,mc,pdyn,cinf,precTS,particles,leftover,drained,6.,splitfac=4,prec_2D=prec2D,maccoat=macscale,saveDT=saveDT,clogswitch=clogswitch,infilt_method=infiltmeth,exfilt_method=exfiltmeth,film=film,infiltscale=infiltscale)

            if hdf5pick:
                with h5py.File(mc.stochsoil, 'r+') as f:
                    dset = f["theta"]
                    dset[:,:,i]=np.reshape((mc.soilmatrix.loc[mc.soilgrid.ravel()-1,'tr']+(mc.soilmatrix.ts-mc.soilmatrix.tr)[mc.soilgrid.ravel()-1]*thS.ravel()*0.01).values,np.shape(thS))
            
                particles.to_hdf('./results/P_'+runname+'.h5','table')
    
                with h5py.File('./results/S_'+runname+'.h5', 'r+') as f:
                    dset = f["states"]
                    dset = [t, len(particles), leftover, len(drained), ix]
            else:
                TSstore[i,:,:]=rE.part_store(particles,mc)
                thetastore[i,:,:]=np.reshape((mc.soilmatrix.loc[mc.soilgrid.ravel()-1,'tr']+(mc.soilmatrix.ts-mc.soilmatrix.tr)[mc.soilgrid.ravel()-1]*thS.ravel()*0.01).values,np.shape(thS))
                with open(''.join([wdir,'/results/Z',runname,'_Mstat.pick']),'wb') as handle:
                   pickle.dump(pickle.dumps([pickle.dumps(particles),pickle.dumps([leftover,drained,t,TSstore,thetastore,npart,i])]), handle, protocol=2)
