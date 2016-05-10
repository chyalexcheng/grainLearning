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
fout.write(' '.join([' E_50','v_50','phi','\n']))

# loop over particle filter
for i in xrange(2000):
   data = np.load('mcSimulations/'+law+'/'+'%d/'%num+'%2.1f/'%conf+str(i+1)+'.npy').item()
   e_r = [0]+list(0.5*(np.array(data['e_v'])-np.array(data['e_a'])))
   e_a = [0]+data['e_a']
   devSig = [0]+list((np.array(data['s33_over_s11'])-1)*conf*1e6)

   ## calculate macro-parameters
   sigRatioMax = max(data['s33_over_s11'])
   
   # friction angle
   frictAngle = degrees(asin((sigRatioMax-1)/(sigRatioMax+1)))
   
   # E_50 and v_50
   devSig_max = max(devSig)
   devSig_50 = .5*devSig_max
   dDevSig = []
   for sig in devSig:
		if sig<devSig_50:	dDevSig.append(sig-devSig_50)
		else: break

	# linear interpolation
   n50 = len(dDevSig)-1
   e_a_50 = e_a[n50]+(e_a[n50+1]-e_a[n50])/(devSig[n50+1]-devSig[n50])*(devSig_50-devSig[n50])
   e_r_50 = e_r[n50]+(e_r[n50+1]-e_r[n50])/(devSig[n50+1]-devSig[n50])*(devSig_50-devSig[n50])
   E_50 = devSig_50/(e_a_50*0.01)
   v_50 = -e_r_50/e_a_50
   
   # write macro-parameters
   fout.write('%15.5e'%E_50+'%15.5e'%v_50+'%15.5e'%frictAngle+'\n')
