#CAOS EFU Model Parameter File
#timing and files

wd='.'                      #working_directory
precf='irr_specht.dat'    #top_boundary_flux_precip
etf='etp.dat'               #top_boundary_flux_etp
inimf='05moist.dat'  #initial_moisture
outdir='out'                #output_directory [not used while testing]
t_end=86400                   #end_time[s] [not used while testing]
t_out=60
part_sizefac=500             #particle_sized_definition_factor
grid_sizefac=0.01           #grid_size_definition_factor [m]
subsfac=10                  #subsampling_rate [percent] [deprecated]
macscalefac=100             #scaling factor converting macropore space and particle size 
t_dini=5.                   #initial_time_step[s]
t_dmx=12.                   #maximal_time_step[s]
t_dmn=0.01                  #minimal_time_step[s]
refarea=1.                  #reference_area_of_obs[m2]
soildepth=-0.7              #depth_of_soil_column[m]
smooth=(3,3)                #smoothing window for thS calculations as no. of cells

#macropore 
macbf='macbase.dat'         #macropore definition file [deprecated]
macdef='05mc.dat'        #macropore definition file > needed if nomac==False
macimg='wollef_macimages.dat'         #macropore image list file > needed if nomac==Image
nomac=False#'Image'   		#switch of macropores 'True, False, 'Single','Image''
mxwidth=0.35                #width of domain
domain_safety=False			#adjust domain width in case of deviation from assumed optimal width (if True)

#bromid tracer data [deprecated]
tracerbf='brspecht.dat'       #tracer base file
tracer_t_ex=8600.0          #time to excavation   
tracer_horgrid=0.1          #gridspacing horizontally   
tracer_vertgrid=0.1         #gridspacing vertically

tracer_site=97              #site ID of sprinkler experiment
tracer_CI=25.43             #
tracer_SD_CI=2.1            #
tracer_appl_Br=4.19595      #applied mass of bromide
tracer_SD_Br=0.3465         #
tracer_time=2.3             #
tracer_intensity=11.05652174#
tracer_c_br=0.165           #

#soil matrix properties
matrixbf='matrix_weierbach.dat'       #matrix base file
matrixdeffi='matrixdef_weierbach.dat' #matrix definition file

#resolution BB stain image
stain_res=0.001             #[m/px] 1px = 1mm
