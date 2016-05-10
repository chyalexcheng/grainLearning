# encoding: utf-8
# get macro-parameters from stress-strain relationships

import numpy as np
from math import asin
from math import degrees

num = input("Number of particles:")
conf = input("Confining pressure in MPa:")
nstep = input("Number of PF steps:")
law = input("Contact law: 1) Hertz-Mindlin or 2) Cundall Linear")
if law == 1: law = 'HM'
if law == 2: law = 'CL'

# write macro-parameters corresponding to each set of micro-parameters (particle set)
fout = file('%d/'%num+law+'/'+'%2.1f/'%conf+'macroParams.dat','w')
fout.write(' '.join(['E','v','E_50','v_50','phi','psi','\n']))

# loop over particle filter
for i in range(2000):
   data = np.load('mcSimulations/'+law+'/'+'%d/'%num+'%2.1f/'%conf+str(i+1)+'.npy').item()
   e_r = 0.5*(np.array(data['e_v'])-np.array(data['e_a']))
   e_a = data['e_a']
   # calculate macro-parameters
   sigRatioMax = max(data['s33_over_s11'])
   # friction angle
   frictAngle = degrees(asin((sigRatioMax-1)/(sigRatioMax+1)))
   # E and v
   devSig_0 = (data['s33_over_s11'][0]-1)*conf*1e6
   e_a_0 = e_a[0]
   e_r_0 = e_r[0]
   E_0 = devSig_0/(e_a_0*0.01)
   v_0 = -e_r_0/e_a_0
   # E_50 and v_50
   devSig_max = (max(data['s33_over_s11'])-1)*conf*1e6
   devSig_50 = .5*devSig_max
   sigRatio_50 = (conf*1e6+devSig_50)/(conf*1e6)
   dSigRatioABS = [abs(sr-sigRatio_50) for sr in data['s33_over_s11']]
   n50 = dSigRatioABS.index(min(dSigRatioABS))
   e_a_50 = e_a[n50]
   e_r_50 = e_r[n50]
   E_50 = devSig_50/(e_a_50*0.01)
   v_50 = -e_r_50/e_a_50
   # dilatancy angle

   de_r = [e_r[j+1]-e_r[j] for j in range(len(e_r)-1)]
   de_a = [e_a[j+1]-e_a[j] for j in range(len(e_a)-1)]
   epsRatio = np.array(de_a)/np.array(de_r)
   epsRatioMax = -1e4
   for j in range(len(epsRatio)):
      if epsRatio[j]>epsRatioMax and -1<epsRatio[j]<0:
         epsRatioMax = epsRatio[j]
   dilaAngle = degrees(asin((1+epsRatioMax)/(1-epsRatioMax)))
   # write macro-parameters
   fout.write('%15.5e'%E_0+'%15.5e'%v_0+'%15.5e'%E_50+'%15.5e'%v_50+'%15.5e'%frictAngle+'%15.5e'%dilaAngle+'\n')
