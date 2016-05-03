# encoding: utf-8

import numpy as np

num = input("Number of particles:")
conf = input("Confining pressure in MPa:")
nstep = input("Number of PF steps:")
law = input("Contact law: 1) Hertz-Mindlin or 2) Cundall Linear")
if law == 1: law = 'HM'
if law == 2: law = 'CL'

# bundle MC simulation results in one data file
fout = file('%d/'%num+law+'/'+'%2.1f/'%conf+'MCS.dat','w')

# loop over particle filter
for i in range(2000):
   fout.write(str(i+1)+'\n')
   data = np.load('mcSimulations/'+law+'/'+'%d/'%num+'%2.1f/'%conf+str(i+1)+'.npy').item()
   # loop over simulation step
   # use radial strain, stress ratio and volumetric strain as state variables
   for j in range(nstep):
      fout.write('%15.5e'%(0.5*(data['e_v'][j]-data['e_a'][j]))+'%15.5e'%data['s33_over_s11'][j]+'%15.5e'%data['e_v'][j]+'\n')
