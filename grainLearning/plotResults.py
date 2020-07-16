""" Author: Hongyang Cheng <chyalexcheng@gmail>
	 Plot results of parameter identification
	 1) evolution of weighted averages of identified parameters versus time
	 2) probability density function of identified parameters at the final step
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from os import system, listdir, path
from matplotlib.offsetbox import AnchoredText
from scipy import stats

params = {'lines.linewidth': 1,'backend': 'ps','axes.labelsize': 12,'font.size': 12, 'legend.fontsize': 9,'xtick.labelsize': 9,'ytick.labelsize': 9,'text.usetex': True,'font.family': 'serif','legend.edgecolor': 'k','legend.fancybox':False}
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams.update(params)

# get weight from files
def getWeight(wName):
	# get weight
	wFile = open(wName, "r")
	wLines = wFile.read().splitlines()
	wFile.close(); 	wData =[]
	# extract weights at specified step
	for i in range(len(wLines)): wData.append(np.float64(wLines[i].split()))
	return wData

def plotIPs(names,ips,covs,nStep,weight,params):
	# plot weighted average for each parameter
	num = len(names)
	nCols = np.ceil(num/2)
	plt.figure('Posterior means over parameters')
	for i in range(num):
		plt.subplot(2,nCols,i+1)
		plt.plot(ips[:,i])
		plt.xlabel('loading step')
		plt.ylabel(r'$|'+names[i]+r'|$')
		plt.grid(True)
	plt.tight_layout()

	# plot coefficient of variance for each parameter
	plt.figure('Posterior coefficients of variance over parameters')
	for i in range(num):
		plt.subplot(2,nCols,i+1)
		plt.plot(covs[:,i])
		plt.xlabel('loading step')
		plt.ylabel(r'$COV('+names[i]+')$')
		plt.grid(True)
	plt.tight_layout()

	# plot probability density function of identified parameters
	for i,name in enumerate(names):
		plt.figure('PDF of '+name)
		for j in range(6):
			plt.subplot(2,3,j+1)
			plt.plot(params[:,i],weight[:,int(nStep*(j+1)/6-1)],'o')
			plt.title('NO.%3i loading step'%(int(nStep*(j+1)/6-1)))
			plt.xlabel(r'$'+name+'$')
			plt.ylabel('Posterior PDF')
			plt.grid(True)
		plt.tight_layout()

	plt.show()


def plotAllSamples(smcSamples, names):
	num = len(names)
	nPanels = np.ceil(num / 2)
	numOfIters = len(smcSamples)
	plt.figure('Resampled parameter space')
	for n in range(int(nPanels)):
		plt.subplot(1, nPanels, n + 1)
		for i in range(numOfIters):
			plt.plot(smcSamples[i][:, 2 * n], smcSamples[i][:, 2 * n + 1], 'o', label='iterNO. %.2i' % i)
			plt.xlabel(r'$' + names[2 * n] + '$')
			plt.ylabel(r'$' + names[2 * n + 1] + '$')
			plt.legend()
		plt.legend()
		plt.tight_layout()
	plt.show()

def numAndExpData(numFiles, p0, q0, n0, e_a0, e_r0):

	e_v0=np.array(e_a0)+2.*np.array(e_r0)
	e_a1,e_r11,e_r21,n1,q1,p1 = np.genfromtxt(numFiles[0]).transpose(); e_v1=e_a1+e_r11+e_r21
	e_a2,e_r12,e_r22,n2,q2,p2 = np.genfromtxt(numFiles[1]).transpose(); e_v2=e_a2+e_r12+e_r22
	e_a3,e_r13,e_r23,n3,q3,p3 = np.genfromtxt(numFiles[2]).transpose(); e_v3=e_a3+e_r13+e_r23

	plt.figure(1)
	plt.plot(e_a0, 'bo', e_a1, '-', e_a2, '--', e_a3, '-.')
	plt.xlabel('Simulation step', fontsize=18)
	plt.ylabel('Axial strain (\%)', fontsize=18)
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.grid(True)
	plt.savefig('e_a.png')

	plt.figure(2)
	plt.plot(e_v0, 'bo', e_v1, '-', e_v2, '--', e_v3, '-.')
	plt.xlabel('Simulation step', fontsize=18)
	plt.ylabel('Volumetric strain', fontsize=18)
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.grid(True)
	plt.savefig('e_v.png')

	plt.figure(3)
	plt.plot(n0, 'bo', n1, '-', n2, '--', n3, '-.')
	plt.xlabel('Simulation step', fontsize=18)
	plt.ylabel('Porosity', fontsize=18)
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.grid(True)
	plt.savefig('n.png')

	plt.figure(4)
	plt.plot(p0, 'bo', p1, '-', p2, '--', p3, '-.')
	plt.xlabel('Simulation step', fontsize=18)
	plt.ylabel('Mean stress (MPa)', fontsize=18)
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.grid(True)
	plt.savefig('p.png')

	plt.figure(5)
	plt.plot(q0, 'bo', q1, '-', q2, '--', q3, '-.')
	plt.xlabel('Simulation step', fontsize=18)
	plt.ylabel('Deviatoric stress', fontsize=18)
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.grid(True)
	plt.savefig('q.png')


def plotExpAndNum(name, names, iterNO, weight, mcFiles, numFiles, label1, label2, label3, label4, p, q, n, e_a, e_r):
	params = {'lines.linewidth': 1.0,'backend': 'ps','axes.labelsize': 10,'font.size': 10, 'legend.fontsize': 9,'xtick.labelsize': 9,'ytick.labelsize': 9,'text.usetex': True,'font.family': 'serif','legend.edgecolor': 'k','legend.fancybox':False}
	matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
	matplotlib.rcParams.update(params)
	titles = ['(a)','(b)']
	if name[-2:] == 'I2': e_a = (np.array(e_a)+2*np.array(e_r))/3.

	# get weight and turning points in the graphs
	nSample, numOfObs = weight.shape
	turns = [1,30,56,80,numOfObs]

	# prepare ensembles
	enP = np.zeros([nSample, numOfObs])
	enQ = np.zeros([nSample, numOfObs])
	enN = np.zeros([nSample, numOfObs])
	enC = np.zeros([nSample, numOfObs])
	for i in range(nSample):
		C,CN,K0,e_r11,e_r21,e_a1,n1,overlap,p1,q1 = np.genfromtxt(mcFiles[i]).transpose();
		enP[i,:] = p1; enQ[i,:] = q1; enN[i,:] = n1; enC[i,:] = CN
	# get ensemble average
	enAvgP = np.zeros(numOfObs); enStdP = np.zeros(numOfObs);
	enAvgQPRatio = np.zeros(numOfObs); enStdQPRatio = np.zeros(numOfObs);
	enAvgN = np.zeros(numOfObs); enStdN = np.zeros(numOfObs);
	enAvgC = np.zeros(numOfObs); enStdC = np.zeros(numOfObs);
	for i in range(numOfObs):
		enAvgP[i] = enP[:,i].dot(weight[:,i])
		enAvgQPRatio[i] = (enQ/enP)[:,i].dot(weight[:,i])
		enAvgN[i] = enN[:,i].dot(weight[:,i])
		enAvgC[i] = enC[:,i].dot(weight[:,i])
	# get diagonal variance
	for i in range(numOfObs):
		enStdP[i] = ((enP[:,i]-enAvgP[i])**2).dot(weight[:,i])
		enStdQPRatio[i] = (((enQ/enP)[:,i]-enAvgQPRatio[i])**2).dot(weight[:,i])
		enStdN[i] = ((enN[:,i]-enAvgN[i])**2).dot(weight[:,i])
		enStdC[i] = ((enC[:,i]-enAvgC[i])**2).dot(weight[:,i])
	# get standard deviation
	enStdP = np.sqrt(enStdP)
	enStdQPRatio = np.sqrt(enStdQPRatio)
	enStdN = np.sqrt(enStdN)
	enStdC = np.sqrt(enStdC)

	fig = plt.figure(figsize=(20/2.54,5/2.54))
	ax = fig.add_subplot(1, 2, 1)
	lines = []; ls = ['-', '-.', ':','-']
	for i in range(len(numFiles)):
		C,CN,K0,e_r11,e_r21,e_a1,n1,overlap,p1,q1 = np.genfromtxt(numFiles[i]).transpose();
		e_v1=e_a1+e_r11+e_r21
		l2, = ax.plot(100*e_a1, q1/p1,label=r'$E_c$'+'=%.1f GPa, '%(label1[i]/1e9)+r'$\mu$'+'=%.3f, '%label2[i]+r'$k_m$'+'=%.3f$\times10^{-3}$ N$\cdot$mm, '%(label3[i]/1e3)+r'$\eta_m$'+'=%.3f'%label4[i],ls = ls[i],color = 'k')
		lines.append(l2)
	l1, = ax.plot(e_a, np.array(q)/np.array(p), 'o',color='darkblue',ms=2.5); lines.append(l1)
	l0, = ax.plot(e_a, enAvgQPRatio, '-',color='darkred'); lines.append(l0)
	lowBound = enAvgQPRatio-2*enStdQPRatio; upBound = enAvgQPRatio+2*enStdQPRatio
	for front,back in zip(turns[:-1],turns[1:]):
		ax.fill_between(e_a[front-1:back], lowBound[front-1:back], upBound[front-1:back], color = 'darkred', alpha = 0.3, linewidth=0.0)
	anchored_text = AnchoredText(titles[0], loc=2, frameon=False, pad=-0.3)
	ax.add_artist(anchored_text)
	ax.set_xlabel(r'$\varepsilon_a$ (\%)')
	ax.set_ylabel(r'$q/p$')
	ax.set_xlim(xmin=0)
	ax.grid(True)

	ax = fig.add_subplot(1, 2, 2)
	for i in range(len(numFiles)):
		C,CN,K0,e_r11,e_r21,e_a1,n1,overlap,p1,q1 = np.genfromtxt(numFiles[i]).transpose();
		e_v1=e_a1+e_r11+e_r21
		l2, = ax.plot(1-n1[1:],p1[1:],label=r'$E_c$'+'=%.1f GPa, '%(label1[i]/1e9)+r'$\mu$'+'=%.3f, '%label2[i]+r'$k_m$'+'=%.3f$\times10^{-3}$ N$\cdot$mm, '%(label3[i]/1e3)+r'$\eta_m$'+'=%.3f'%label4[i],ls = ls[i],color = 'k')
	l1, = ax.plot(1-np.array(n), p, 'o',color='darkblue',label='Experimental data',ms=2.5)
	l0, = ax.plot(1-enAvgN, enAvgP, '-',color='darkred')
	lowBound = enAvgP-2*enStdP; upBound = enAvgP+2*enStdP
	for front,back in zip(turns[:-1],turns[1:]):
		ax.fill_between(1-enAvgN[front-1:back], lowBound[front-1:back], upBound[front-1:back], color = 'darkred', alpha = 0.3, linewidth=0.0)
	anchored_text = AnchoredText(titles[1], loc=2, frameon=False, pad=-0.3)
	ax.add_artist(anchored_text)
	ax.set_ylabel(r'$p$ (MPa)')
	ax.set_xlabel(r'1$-n$')
	ax.set_xticks([0.61,0.615,0.62])
	ax.set_ylim(ymin=0)
	ax.grid(True)
	
	fig.legend(tuple(lines),tuple([r'$E_c$'+'=%.1f GPa, '%(label1[i]/1e9)+r'$\mu$'+'=%.3f\n'%label2[i]+r'$k_m$'+r'=%.3f$\times10^{-3}$ N$\cdot$mm'%(label3[i]/1e3)+'\n'+r'$\eta_m$'+'=%.3f'%label4[i] for i in range(len(numFiles))]+['Experimental data']+['Posterior ensemble']),loc = 'right',ncol=1,handlelength=1.45,labelspacing=0.8,frameon=False)
	fig.subplots_adjust(left=0.06, bottom=0.20, right=0.74, top=0.98, hspace=0, wspace=0.35)
	plt.savefig('expAndDEM'+iterNO+'.pdf')
	plt.show()

	return enAvgP, enStdP, enAvgQPRatio, enStdQPRatio, enAvgN, enStdN, enAvgC, enStdC

def plotExpSequence(name, names, varsDir, numFiles, label1, label2, label3, label4, p, q, n, e_a, e_r):

	for i in range(55):
		titles = ['(a)','(b)']
		if name[-2:] == 'I2': e_a = (np.array(e_a)+2*np.array(e_r))/3.
		fig = plt.figure(figsize=(15/2.54,6.25/2.54))
		ax = fig.add_subplot(1, 2, 1)
		l1, = ax.plot(e_a[:56], np.array(q[:56])/np.array(p[:56]), '-',color='gray')
		ax.plot(e_a[i], np.array(q[i])/np.array(p[i]), '^',color='k')
		lines = [l1]; ls = ['-', '--', '-.', ':']
		anchored_text = AnchoredText(titles[0], loc=2, frameon=False, pad=-0.3)
		ax.add_artist(anchored_text)
		ax.set_xlabel(r'$\varepsilon_a$ (\%)')
		ax.set_ylabel(r'$q/p$')
		ax.set_xlim(xmin=0)
		ax.grid(True)
		ax = fig.add_subplot(1, 2, 2)
		l1, = ax.plot(p[:56], n[:56], '-',color='gray',label='Experimental data',ms=4)
		ax.plot(p[i], n[i], '^',color='k')
		anchored_text = AnchoredText(titles[1], loc=2, frameon=False, pad=-0.3)
		ax.add_artist(anchored_text)
		ax.set_xlabel(r'$p$ (MPa)')
		ax.set_ylabel(r'$n$')
		ax.set_xlim(xmin=0)
		ax.grid(True)
		fig.subplots_adjust(left=0.09, bottom=0.18, right=0.98, top=0.98, hspace=0, wspace=0.35)
		#~ plt.savefig('%02d.png'%i,dpi=300)
		plt.show()

def plotExpAndNumHalfPage(name, names, varsDir, numFiles, label1, label2, label3, label4, p, q, n, e_a, e_r):
	params = {'lines.linewidth': 1.5,'backend': 'ps','axes.labelsize': 17.7,'font.size': 17.7, 'legend.fontsize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15,'text.usetex': True,'font.family': 'serif','legend.edgecolor': 'k','legend.fancybox':False}
	matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
	matplotlib.rcParams.update(params)
	titles = ['(a)','(b)']
	if name[-2:] == 'I2': e_a = (np.array(e_a)+2*np.array(e_r))/3.
	fig = plt.figure(figsize=(16/2.54,14/2.54))

	ax = fig.add_subplot(2, 1, 1)
	l1, = ax.plot(e_a, np.array(q)/np.array(p), 'o',color='darkred')
	lines = [l1]; ls = ['-', '--', '-.', ':']
	for i in range(len(numFiles)):
		C,CN,K0,e_r11,e_r21,e_a1,n1,overlap,p1,q1 = np.genfromtxt(numFiles[i]).transpose();
		e_v1=e_a1+e_r11+e_r21
		l2, = ax.plot(100*e_a1, q1/p1,label=r'$E_c$'+'=%.1f, '%(label1[i]/1e9)+r'$\mu$'+'=%.3f, '%label2[i]+r'$k_m$'+'=%.3f, '%(label3[i]/1e3)+r'$\eta_m$'+'=%.3f'%label4[i],ls = ls[i],color = 'k')
		lines.append(l2)
		anchored_text = AnchoredText(titles[0], loc=2, frameon=False, pad=-0.3)
		ax.add_artist(anchored_text)
		ax.set_xlabel(r'$\varepsilon_a$ (\%)')
		ax.set_ylabel(r'$q/p$')
		ax.set_xlim(xmin=0)
		ax.grid(True)

	ax = fig.add_subplot(2, 1, 2)
	l1, = ax.plot(p, n, 'o',color='darkred',label='Experimental data')
	for i in range(len(numFiles)):
		C,CN,K0,e_r11,e_r21,e_a1,n1,overlap,p1,q1 = np.genfromtxt(numFiles[i]).transpose();
		e_v1=e_a1+e_r11+e_r21
		l2, = ax.plot(p1[1:], n1[1:],label=r'$E_c$'+'=%.1f, '%(label1[i]/1e9)+r'$\mu$'+'=%.3f, '%label2[i]+r'$k_m$'+'=%.3f, '%(label3[i]/1e3)+r'$\eta_m$'+'=%.3f'%label4[i],ls = ls[i],color = 'k')

	anchored_text = AnchoredText(titles[1], loc=2, frameon=False, pad=-0.3)
	ax.add_artist(anchored_text)
	ax.set_xlabel(r'$p$ (MPa)')
	ax.set_ylabel(r'$n$')
	ax.set_xlim(xmin=0)
	ax.grid(True)

	fig.legend(tuple(lines),tuple(['Experimental\ndata']+[r'$E_c$'+'=%.1f\n'%(label1[i]/1e9)+r'$\mu$'+'=%.3f\n'%label2[i]+r'$k_m$'+'=%.3f\n'%(label3[i]/1e3)+r'$\eta_m$'+'=%.3f'%label4[i] for i in range(len(numFiles))]),loc = 'right',nCols=1,handlelength=1.45,labelspacing=1.5,frameon=False)
	fig.subplots_adjust(left=0.13, bottom=0.105, right=0.66, top=0.98, hspace=0.4, wspace=0.35)
	#~ plt.savefig('expAndDEM'+varsDir[-2]+'.png',dpi=600); plt.savefig('expAndDEM'+varsDir[-2]+'.tif',dpi=600); plt.savefig('expAndDEM'+varsDir[-2]+'.pdf',dpi=600);
	plt.show()

def microMacroPDF(name, step, pData, varsDir, weight, mcFiles, loadWeights=False):
	import seaborn as sns
	from resample import unWeighted_resample
	params = {'lines.linewidth': 1.5,'backend': 'ps','axes.labelsize': 18,'font.size': 18, 'legend.fontsize': 16,'xtick.labelsize': 16,'ytick.labelsize': 16,'text.usetex': True,'font.family': 'serif','legend.edgecolor': 'k','legend.fancybox':False}
	matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
	matplotlib.rcParams.update(params)

	# read in the original sampled parameter set
	Xlabels = [r'$E_c$'+r' $(\mathrm{GPa})$', r'$\mu$', r'$k_m$'+r' $(\mu\mathrm{Nm)}$', r'$\eta_m$']
	Ylabels = [r'$p$'+r' $(\mathrm{MPa})$', r'$q/p$', r'$n$', r'$C^*$']
	if loadWeights: wPerPair = np.load(varsDir+'/wPerPair%i.npy'%step).item()
	else: wPerPair = {}

	# Monte Carlo sequence
	nSample, numOfObs = weight.shape
	enQOIs = np.zeros([4,nSample])
	for i in range(nSample):
		C,CN,K0,e_r11,e_r21,e_a1,n1,overlap,p1,q1 = np.genfromtxt(mcFiles[i]).transpose();
		enQOIs[0,i] = p1[step]; enQOIs[1,i] = q1[step]/p1[step]; enQOIs[2,i] = n1[step]; enQOIs[3,i] = CN[step]
	# importance resampling
	ResampleIndices = unWeighted_resample(weight[:,step],nSample*10)
	particles = np.copy(pData)
	particles[0,:] /= 1e9; particles[2,:] /= 1e3
	particles = particles[:,ResampleIndices]
	enQOIs = enQOIs[:,ResampleIndices]
	# remove porosity as QOI if iterNO > 0
	if 'iterPF0' not in mcFiles[0]:
		enQOIs = np.vstack([enQOIs[:2,:],enQOIs[-1,:]])
		for i in [9,10,11,12]: wPerPair[i]=wPerPair[i+4]
		Ylabels = [r'$p$'+r' $(\mathrm{MPa})$', r'$q/p$', r'$C^*$']

	# plot figures
	figNO = 0
	fig = plt.figure('microMacroUQ_iterPF'+varsDir[-1]+'_%i'%step, figsize=(12/2.54,12/2.54))
	sns.set(style="ticks",rc=matplotlib.rcParams)
	for i in range(enQOIs.shape[0]):
		for j in range(particles.shape[0]):
			figNO += 1
			# plot histograms at the initial and final steps
			ax = fig.add_subplot(enQOIs.shape[0],4,figNO)
			cmap = sns.dark_palette("black", as_cmap=True)
			#~ ax = sns.kdeplot(,,cmap=cmap,kernel='gau',cut=3,n_levels=20,bw='silverman',linewidths=0.8); ax.grid()
			if loadWeights:
				minScore, maxScore, p0, w0 = wPerPair[figNO]
			else:
				data = np.array([particles[j],enQOIs[i]])
				pMin = [min(particles[j]),min(enQOIs[i])]
				pMax = [max(particles[j]),max(enQOIs[i])]
				minScore, maxScore, p0, w0 = getLogWeightFromGMM(data.T,pMin,pMax,1e-2);
				wPerPair[figNO] = (minScore, maxScore, p0, w0)
			X,Y = p0
			plt.contour(X, Y, w0, cmap=cmap,levels=np.linspace(minScore-abs(minScore)*0.1,maxScore,10),linewidths=0.7); plt.grid()
			ax.locator_params(axis='both',nbins=2)
			ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
			if i == enQOIs.shape[0]-1:
				ax.set_xlabel(Xlabels[j],size = params['font.size'],labelpad=10); ax.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off', left='off', labelleft='off', labelsize = params['xtick.labelsize'])
			if j == 0:
				ax.set_ylabel(Ylabels[i],size = params['font.size'],labelpad=10); ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='on', labelleft='on', labelsize = params['xtick.labelsize'])
			if i==enQOIs.shape[0]-1 and j==0:
				ax.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off', left='on', labelleft='on', labelsize = params['xtick.labelsize'])
			if i!=enQOIs.shape[0]-1 and j!=0:
				ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
			xMin, xMax = ax.get_xlim()
			yMin, yMax = ax.get_ylim()
			dx = (xMax-xMin)/4; dy = (yMax-yMin)/4;
			ax.set_xticks(np.round([xMin+dx,xMax-dx],int(np.ceil(-np.log10(dx)))))
			ax.set_yticks(np.round([yMin+dy,yMax-dy],int(np.ceil(-np.log10(dy)))))
	if not loadWeights: np.save(varsDir+'/wPerPair%i.npy'%step,wPerPair)
	plt.tight_layout()
	fig.subplots_adjust(left=0.21, bottom=0.16, right=0.99, top=0.99, hspace=0., wspace=0.)
	plt.savefig('microMacroUQ_iterPF'+varsDir[-1]+'_%i.pdf'%step,dpi=600);
	#~ plt.show()
	return wPerPair

# kde estimation
def getPDF(pDataResampled,pMin,pMax):
	kde = stats.gaussian_kde(pDataResampled)
	pDataNew = np.linspace(pMin,pMax,2000)
	wDataNew = kde(pDataNew)
	wDataNew[0], wDataNew[-1] = 0, 0
	return pDataNew, wDataNew

def macroMacroPDF(name, step, pData, varsDir, weight, mcFiles):
	import seaborn as sns
	from resample import unWeighted_resample
	params = {'lines.linewidth': 1.5,'backend': 'ps','axes.labelsize': 20,'font.size': 20, 'legend.fontsize': 18,'xtick.labelsize': 18,'ytick.labelsize': 18,'text.usetex': True,'font.family': 'serif','legend.edgecolor': 'k','legend.fancybox':False}
	matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
	matplotlib.rcParams.update(params)

	# read in the original sampled parameter set
	labels = [r'$p$'+r' $(\mathrm{MPa})$', r'$q/p$', r'$n$', r'$C^*$']

	# Monte Carlo sequence
	nSample, numOfObs = weight.shape
	enQOIs = np.zeros([4,nSample])
	for i in range(nSample):
		C,CN,K0,e_r11,e_r21,e_a1,n1,overlap,p1,q1 = np.genfromtxt(mcFiles[i]).transpose();
		enQOIs[0,i] = p1[step]; enQOIs[1,i] = q1[step]/p1[step]; enQOIs[2,i] = n1[step]; enQOIs[3,i] = CN[step]
	# importance resampling
	ResampleIndices = unWeighted_resample(weight[:,step],nSample*10)
	sortedIndex = np.argsort(weight[:,step])
	particles = np.copy(pData)
	particles[0,:] /= 1e9; particles[2,:] /= 1e3
	particles = particles[:,ResampleIndices]
	enQOIs = enQOIs[:,ResampleIndices]

	# plot figures
	figNO = 0
	fig = plt.figure(figsize=(18/2.54,18/2.54))
	sns.set(style="ticks",rc=matplotlib.rcParams)
	for i in range(4):
		for j in range(4):
			figNO += 1
			# plot histograms at the initial and final steps
			ax = fig.add_subplot(4, 4, figNO)
			if i == j:
				# estimate pdf using gaussian kernals
				p0, w0 = getPDF(enQOIs[i,:],min(enQOIs[i,:]),max(enQOIs[i,:]))
				ax2 = ax.twinx()
				ax2.fill_between(p0,w0,alpha=0.63,facecolor='black')
				ax2.tick_params(axis='y', right='off', labelright='off')
				ax.set_ylim(min(p0),max(p0));
			elif j<i:
				cmap = sns.dark_palette("black", as_cmap=True)
				ax = sns.kdeplot(enQOIs[j],enQOIs[i],cmap=cmap,kernel='gau',n_levels=20,bw='silverman',linewidths=0.8); ax.grid()
				ax.locator_params(axis='both',tight=True,nbins=2)
				ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
				ax.set_xlim(min(enQOIs[j,:]),max(enQOIs[j,:])); ax.set_ylim(min(enQOIs[i,:]),max(enQOIs[i,:]))
			elif j>i:
				cmap = sns.dark_palette("black", as_cmap=True)
				ax.scatter(enQOIs[j,sortedIndex],enQOIs[i,sortedIndex],s=25,c=weight[sortedIndex,step],cmap='Greys'); ax.grid()
				ax.locator_params(axis='both',tight=True,nbins=2)
				ax.set_xlim(min(enQOIs[j,:]),max(enQOIs[j,:])); ax.set_ylim(min(enQOIs[i,:]),max(enQOIs[i,:]))
			if i == 3:
				ax.set_xlabel(labels[j],size = params['font.size'],labelpad=10); ax.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off', left='off', labelleft='off', labelsize = params['xtick.labelsize'])
			if j == 0:
				ax.set_ylabel(labels[i],size = params['font.size'],labelpad=10); ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='on', labelleft='on', labelsize = params['xtick.labelsize'])
			if i==3 and j==0:
				ax.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off', left='on', labelleft='on', labelsize = params['xtick.labelsize'])
			if i!=3 and j!=0:
				ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

	plt.tight_layout()
	fig.subplots_adjust(left=0.165, bottom=0.12, right=0.98, top=0.99, hspace=0.1, wspace=0.1)
	plt.savefig('macroMacroUQ_'+varsDir[-8:-1]+'_%i.pdf'%step,dpi=600);
	plt.show()

def plot3DScatter(xName,yName, zName, x,y,z):
	from mpl_toolkits.mplot3d import Axes3D

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d'); ax.scatter(x,y,z)
	ax.set_xlabel(xName); ax.set_ylabel(yName); ax.set_zlabel(zName)
	#~ ax.set_zlim([-5-0.1,-5+0.1])
	#~ ax.set_xlim([0.3,0.4])
	plt.show()


def polySmooth(y):
	x = np.arange(len(y))
	yhat = savitzky_golay(y, 51, 10) # window size 51, polynomial order 3
	return yhat

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
