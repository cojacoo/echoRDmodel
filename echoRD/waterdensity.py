# coding=utf-8
def waterdensity(T,P):
    '''
       Calc density of water depending on T [°C] and P [Pa]
       defined between 0 and 40 °C and given in g/m3
       Thiesen Equation after CIPM
       Tanaka et al. 2001, http://iopscience.iop.org/0026-1394/38/4/3

       NOTE: the effect of solved salts, isotopic composition, etc. remain
       disregarded here. especially the former will need to be closely
       considerd in a revised version! DEBUG.
       
       INPUT:  Temperature T in °C as numpy.array
               Pressure P in Pa as numpy.array (-9999 for not considered)
       OUTPUT: Water Density in g/m3
       
       EXAMPLE: waterdensity(np.array((20,21,42)),np.array(-9999.))
       (cc) jackisch@kit.edu
    '''
    
    import numpy as np
    
    # T needs to be given in °C
    a1 = -3.983035  # °C	 	
    a2 = 301.797    # °C	 	
    a3 = 522528.9   # °C2		
    a4 = 69.34881   # °C	 	
    a5 = 999974.950 # g/m3
    
    dens=a5*(1-((T+a1)**2*(T+a2))/(a3*(T+a4)))
    
    # P needs to be given in Pa
    # use P=-9999 if pressure correction is void
    if P.min()>-9999:
        c1 = 5.074e-10   # Pa-1	 	
        c2 = -3.26e-12   # Pa-1 * °C-1	
        c3 = 4.16e-15    # Pa-1 * °C-2  .
        Cp = 1 + (c1 + c2*T + c3*T**2) * (P - 101325)
        dens=dens*Cp
    
    # remove values outside definition bounds and set to one
    if (((T.min()<0) | (T.max()>40)) & (T.size>1)):
        idx=np.where((T<0) | (T>40))
        dens[idx]=100000.0  #dummy outside defined bounds 
    
    return dens
