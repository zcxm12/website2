##########################################################複製區塊
import matplotlib.pyplot as plt
import time
from scipy import signal
from time import sleep
import sys
import meep as mp
import numpy as np
import random
import multiprocessing as multi
#setup some constants and parameters
M=2**16
tsim=7e-12
eps0 = 8.854187e-12
hbar = 1.05457182e-34
c = 2.99792458e8
kb = 1.380649e-23
hdk=7.63823258e-12
deltap = np.sqrt(1/3/eps0/hbar/c)*kb
T=30000
frq_min = 0.5 #150THz
frq_max = 1.5 #450THz
nfreq = 1000
fcen = (frq_min + frq_max)/2
df = frq_max - frq_min
#define square root of Dn function
def Dnsqt(wla,T):
    return np.sqrt(6*hdk**2*wla/(np.exp(hdk*wla/T)-1)/T**2/np.pi)

#generate an array of the square root of Dn function
Dsqt = []
for n in range(1,M):
    Dsqt.append(Dnsqt(2*np.pi*n/tsim,T))

#define how many times to run to average the results
Ncomp= 100
nfreq =  1000
#define the starting sum of the result
epsilonA = 2
epsilonB = 13
Efsum = np.zeros(nfreq)
Ens = np.zeros(2*M)
resolution = 60
va = 1
dA = 0.2
dB = 0.8
dpml = 1.0
NAB = 3
sz = 2*dpml+(dA+dB)*NAB+2*va
cell = mp.Vector3(0, 0, sz)
pml_layers = [mp.PML(dpml)]



for i in range(Ncomp):
    
    def random_En(t):
        #generate random numbers
        M0 = np.random.normal(0,1)
        Mlp = np.random.normal(0,1,M-1)
        Nlp = np.random.normal(0,1,M-1)
        MM = np.random.normal(0,1)
    
        #generate the array of the fourier transform of E field, we generate l=0 and l=-M separately. 
        #for l=1 to M-1 and l=-1 to -(M-1) are conjutated to each other. 
        #we contruct the whole array by gluing them together by the order from l=0 -> l=M-1 -> l=-M -> l=-(m-1) -> l=-1
        Ef0 = np.array([M0*np.sqrt(6*hdk/np.pi/T)])
        Eflp = np.multiply((Mlp + 1j*Nlp), Dsqt)
        Eflm = np.flip(np.conjugate(Eflp))
        EfM = np.array([MM*Dnsqt(2*np.pi*M/tsim,T)])
        Ef = np.hstack((Ef0,Eflp,EfM,Eflm))
    
        #we normalize Ef and compute its inverse fast fourier transform
        Efn = deltap*T/np.sqrt(tsim)*Ef
        En = np.fft.ifft(Efn)*M
        return En
    En0 = random_En(1)
    En1 = random_En(1)
    #generate some feedback to check the progress of the loop
    x = i/Ncomp*100
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('a'*int(x), int(x)))
    sys.stdout.flush()
    sleep(0.0001)
    
    
    def source0(t):
        f = int(t*10)
        return En0[f]
    def source1(t):
        f = int(t*10)
        return En1[f]
   


    sources = [mp.Source(mp.CustomSource(src_func=source0),
                     component=mp.Ex,
                     center=mp.Vector3(0,0,random.uniform(-(dA+dB)*NAB/2-va,-(dA+dB)*NAB/2)),
                     ),
              mp.Source(mp.CustomSource(src_func=source1),
                     component=mp.Ex,
                     center=mp.Vector3(0,0,random.uniform((dA+dB)*NAB/2,(dA+dB)*NAB/2+va)),
                     )
              ]
    
    def BlockAB(N, center):
        geometry = []
        for i in range(N):
            geometry = np.append(geometry, mp.Block(mp.Vector3(mp.inf,mp.inf,dA),
                         center=mp.Vector3(z=center-(dA+dB)*N/2 + (dA+dB)*i + dA/2),
                         material=mp.Medium(epsilon=epsilonA)))
            geometry = np.append(geometry, mp.Block(mp.Vector3(mp.inf,mp.inf,dB),
                         center=mp.Vector3(z=center-(dA+dB)*N/2 + (dA+dB)*i + dA + dB/2),
                         material=mp.Medium(epsilon=epsilonB)))
        geometry = geometry.tolist()
        return geometry
    def BlockBA(N, center):
        geometry = []
        geometry = np.append(geometry, mp.Block(mp.Vector3(mp.inf,mp.inf,va),
                         center=mp.Vector3(z=center-(dA+dB)*N/2 - va/2 ),
                         material=mp.Medium(epsilon=1)))
        for i in range(N):
            geometry = np.append(geometry, mp.Block(mp.Vector3(mp.inf,mp.inf,dB),
                         center=mp.Vector3(z=center-(dA+dB)*N/2 + (dA+dB)*i + dB/2),
                         material=mp.Medium(epsilon=epsilonB)))
            geometry = np.append(geometry, mp.Block(mp.Vector3(mp.inf,mp.inf,dA),
                         center=mp.Vector3(z=center-(dA+dB)*N/2 + (dA+dB)*i + dB + dA/2),
                         material=mp.Medium(epsilon=epsilonA)))
        geometry = np.append(geometry, mp.Block(mp.Vector3(mp.inf,mp.inf,va),
                         center=mp.Vector3(z=center + (dA+dB)*N/2 + va/2 ),
                         material=mp.Medium(epsilon=1)))
        geometry = geometry.tolist()
        return geometry
    geometry = BlockBA(3,0)

    
    sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    sources=sources,   
                    dimensions = 1,
                    Courant = 1,
                    resolution=resolution,
                    geometry=geometry)
    
    

    # transmitted flux
    tran_fr = mp.FluxRegion(center=mp.Vector3(0,0,random.uniform(-(dA+dB)*NAB/2,(dA+dB)*NAB/2)))
    tran = sim.add_energy(fcen, df, nfreq, tran_fr)
    
    
    sim.run(until=2090)

    tran_flux = mp.get_magnetic_energy(tran)
    Ts = []
    for i in range(nfreq):
        Ts = np.append(Ts,tran_flux[i])    
    Efsum = Ts
    
    

#average the final result
Efavg = Efsum/Ncomp

#save data to some file or reload file to write more data on it

freqs = np.linspace(1.5e14,4.5e14,1000)
plt.plot(freqs,Efavg*tsim*eps0*4*np.pi*3/M,'g''-')
plt.xlim(1.5e14,4.5e14)
#plt.yscale("log")
#plt.ylim(0.1e-28,400e-28)
plt.grid()
plt.savefig('3AB100_1.png')
plt.show()

np.savez("3AB100_1.npz", Efavg_30000 = Efavg)