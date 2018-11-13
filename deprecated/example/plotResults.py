""" Author: Hongyang Cheng <chyalexcheng@gmail>
	 Plot results of parameter identification
	 1) evolution of weighted averages of identified parameters versus time
	 2) probability density function of identified parameters at the final step
"""

import matplotlib.pyplot as plt
import numpy as np

def plotIPs(names,ips,nSample,weight,params):
	# plot evolution of weighted average of each identified parameter
	plt.figure(1)

	plt.subplot(231)
	plt.plot(ips[:,0])
	plt.xlabel('step')
	plt.ylabel(names[0])
	plt.grid(True)

	plt.subplot(232)
	plt.plot(ips[:,1])
	plt.xlabel('step')
	plt.ylabel(names[1])
	plt.grid(True)

	plt.subplot(233)
	plt.plot(ips[:,2])
	plt.xlabel('step')
	plt.ylabel(names[2])
	plt.grid(True)

	plt.subplot(234)
	plt.plot(ips[:,3])
	plt.xlabel('step')
	plt.ylabel(names[3])
	plt.grid(True)

	plt.subplot(235)
	plt.plot(ips[:,4])
	plt.xlabel('step')
	plt.ylabel(names[4])
	plt.grid(True)
	
	# plot probability density function of identified parameters
	plt.figure(2)
	
	plt.subplot(231)
	plt.bar(np.arange(nSample),weight[:,14])
	plt.title('15 Step')
	plt.xlabel('parameter set numer')
	plt.ylabel('weight')
	plt.grid(True)
	
	plt.subplot(232)
	plt.bar(np.arange(nSample),weight[:,29])
	plt.title('30 Step')
	plt.xlabel('parameter set numer')
	plt.ylabel('weight')
	plt.grid(True)
	
	plt.subplot(233)
	plt.bar(np.arange(nSample),weight[:,44])
	plt.title('45 Step')
	plt.xlabel('parameter set numer')
	plt.ylabel('weight')
	plt.grid(True)
	
	plt.subplot(234)
	plt.bar(np.arange(nSample),weight[:,59])
	plt.title('60 Step')
	plt.xlabel('parameter set numer')
	plt.ylabel('weight')
	plt.grid(True)

	plt.subplot(235)
	plt.bar(np.arange(nSample),weight[:,74])
	plt.title('75 Step')
	plt.xlabel('parameter set numer')
	plt.ylabel('weight')
	plt.grid(True)

	plt.subplot(236)
	plt.bar(np.arange(nSample),weight[:,89])
	plt.title('90 Step')
	plt.xlabel('parameter set numer')
	plt.ylabel('weight')
	plt.grid(True)	
	
	plt.show()
