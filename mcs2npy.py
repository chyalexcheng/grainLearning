# encoding: utf-8

import numpy as np

num = input("Number of particles:")
conf = input("Confining pressure in MPa:")
nstep = input("Number of PF steps:")
law = input("Contact law: 1) Hertz-Mindlin or 2) Cundall Linear")
if law == 1: law = 'HM'
if law == 2: law = 'CL'

# write MCS.data back to binary .npy files in mcSimulations folder
fin = open('%d/'%num+law+'/'+'%2.1f/'%conf+'MCS.dat','r')
lines = fin.readlines()
for i in range(2000):
	n = 101*i+1
	e_a, s33_over_s11, e_v = [], [], []
	for j in range(nstep):
		l = lines[n+j].split('   ')
		e_a.append(float(l[1]))
		e_v.append(float(l[2]))
		s33_over_s11.append(float(l[3]))
	data = {}
	data['e_a'] = e_a; data['s33_over_s11'] = s33_over_s11; data['e_v'] = e_v
	np.save('mcSimulations/'+law+'/'+'%d/'%num+'%2.1f/'%conf+str(i+1)+'.npy',data)
