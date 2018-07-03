import numpy as np

def frange(start,step,end):
    n=int(np.ceil((end-start)/step))+1
    return np.arange(n)*step+start

def stochmod(lz,lx,dz,dx,az,ax,nu,seed=False):
    ''' 
    Create stochastical field
    lx: model length
    lz: model depth
    dx: sampling interval in x direction
    dz: sampling interval in z direction 
    ax: correlation length in x direction
    az: correlation length in z direction
    nu: Hurst exponent
    seed: random number seed (optional)
    example stochmod(100,100,0.1,0.1,2,10,0.5)
    '''

    if (lx >= lz):
      n = int(np.round(lx/float(dx)))
    else:
      n = int(np.round(lz/float(dz)))
    

    h=1 
    while n > (2**h):
      h+=1 
    
    n=2**h

    kxnyq = 1./(2.*dx)
    dkx = 2.*kxnyq/n
    kx = frange(0.,dkx,kxnyq)*2.*np.pi

    kznyq = 1./(2.*dz)
    dkz = 2.*kznyq/n
    kz = np.zeros(n)
    kz[0:n/2+1] = frange(0.,dkz,kznyq)*2.*np.pi
    kz[n/2+1:] = frange(-kznyq+dkz,dkz,-dkz)*2.*np.pi

    ax = np.round(ax/dx)
    az = np.round(az/dz)

    #calculate power spectrum for 1st and 3rd quadrant
    p = np.zeros((n,n/2+1))
    for i in np.arange(n/2+1):
       for j in np.arange(n):
          p[j,i]= (1+(kx[i]*ax)**2+(kz[j]*az)**2)**-(nu+1)

    p = np.sqrt(p)
    if seed:
        np.random.seed(seed)

    p = p * np.exp(-1j*(np.random.uniform(size=(n,n/2+1))-0.5)*2.*np.pi);

    #there are 4 points (dc value and 3 other points) without imaginary parts
    p[0,0]=np.real(p[0,0])
    p[0,n/2]=np.real(p[0,n/2])
    p[n/2,0]=np.real(p[n/2,0])
    p[n/2,n/2]=np.real(p[n/2,n/2])

    #enforce symmetry along top and central horizonal axis by adding columns
    p = np.concatenate([p,np.zeros((n,n-len(kx)))],axis=1)
    p[0,n/2+1:]=np.conj(p[0,1:n/2])[::-1]
    p[n/2,n/2+1:]=np.conj(p[n/2,1:n/2])[::-1]

    #enforce symmetry along left and central vertical axis
    p[n/2+1:,0]=np.flipud(np.conj(p[1:n/2,0]))
    p[n/2+1:,n/2]=np.flipud(np.conj(p[1:n/2,n/2]))

    #enforce symmetries for 2nd and 4th quandrants
    p[1:n/2,n/2+1:]=np.flipud(np.fliplr(np.conj(p[n/2+1:,1:n/2])))
    p[n/2+1:,n/2+1:]=np.flipud(np.fliplr(np.conj(p[1:n/2,1:n/2])))

    s = np.real(np.fft.ifft2(p))
    s=s-np.mean(np.mean(s))
    s=s/np.max(np.max(s))
    s = s/ np.mean(np.std(s))
    s = s[0:int(np.round(lz/dz)+1),0:int(np.round(lx/dx)+1)]

    #Normalization - small grid (Jens):
    s=s-np.mean(np.mean(s))
    s=s/np.max(np.max(s))
    s = s/np.mean(np.std(s))

    return s