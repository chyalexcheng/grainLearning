from smc import *
from plotResults import *
import pickle

# normalized variance parameter
sigma = 1.0; ess = 1.0
obsWeights = [1,1,0.01]
yadeFile = 'mcTriax_e.py'
yadeDataDir = 'iterPFNew1'
obsDataFile = 'obsdata.dat'
obsCtrl = 'e_a'

# ranges of parameters (E, \mu, kr, \mu_r)
paramNames = ['E', 'mu', 'k_r','mu_r']
paramRanges = {'E':[100e9,200e9],'mu':[0.3,0.5],'k_r':[0,1e4],'mu_r':[0.1,0.5]}
numSamples = 61
iterNO = int(yadeDataDir[-1])
sampleDataFile = 'smcTableNew%i.txt'%iterNO
proposalFile = 'gmm_'+yadeDataDir[:-1]+'%i.pkl'%(iterNO-1) if int(yadeDataDir[-1]) != 0 else ''

#~ while abs(ess-0.2)/0.2 > 1e-3:
	#~ # initialize the problem
	#~ smcTest = smc(sigma,obsWeights,yadeFile,yadeDataDir,obsDataFile,obsCtrl)
	#~ smcTest.initialize(paramNames,paramRanges,numSamples,sampleDataFile=sampleDataFile,loadSamples=True,proposalFile=proposalFile)
	#~ # run sequential Monte Carlo; return means and coefficients of variance of PDF over the parameters
	#~ ips, covs = smcTest.run(skipDEM=True)
	#~ # get the parameter samples (ensemble) and posterior probability
	#~ posterior = smcTest.getPosterior()
	#~ smcSamples = smcTest.getSmcSamples()
	#~ # calculate effective sample size
	#~ ess = smcTest.getEffectiveSampleSize()[-1]
	#~ print 'Effective sample size: %f'%ess
	#~ sigma *= 0.99

# initialize the problem
smcTest = smc(sigma,obsWeights,yadeFile,yadeDataDir,obsDataFile,obsCtrl)
smcTest.initialize(paramNames,paramRanges,numSamples,sampleDataFile=sampleDataFile,loadSamples=True,proposalFile=proposalFile)
# run sequential Monte Carlo; return means and coefficients of variance of PDF over the parameters
ips, covs = smcTest.run(skipDEM=True)
# get the parameter samples (ensemble) and posterior probability
posterior = smcTest.getPosterior()
smcSamples = smcTest.getSmcSamples()
# calculate effective sample size
ess = smcTest.getEffectiveSampleSize()[-1]
print 'Effective sample size: %f'%ess

# plot time evolution of effective sample size
plt.plot(smcTest.getEffectiveSampleSize()); plt.show()

for i in range(10): 
	plt.plot(smcTest.getSmcSamples()[0][:,0],smcTest._prior,'o',label='prior');
	plt.plot(smcTest.getSmcSamples()[0][:,0],smcTest._proposal[:,i*10],'o',label='proposal');
	plt.plot(smcTest.getSmcSamples()[0][:,0],smcTest._likelihood[:,i*10],'o',label='likelihood')
	plt.plot(smcTest.getSmcSamples()[0][:,0],smcTest._posterior[:,i*10],'o',label='posterior')
	plt.legend(); plt.show()

#~ # plot means of PDF over the parameters
#~ _ = plotIPs(paramNames,ips.T,covs.T,smcTest.getNumSteps(),posterior,smcSamples[-1])

#~ # resample parameters
#~ caliStep = -1
#~ maxNumComponents = 20
#~ gmm, maxNumComponents = smcTest.resampleParams(caliStep=caliStep,maxNumComponents=maxNumComponents)
#~ 
#~ # plot initial and resampled parameters
#~ plotAllSamples(smcTest.getSmcSamples(),smcTest.getNames())
#~ 
#~ # save trained Gaussian mixture model
#~ gmmList = smcTest.trainGMMinTime(maxNumComponents)
#~ pickle.dump(gmmList, open('gmm_'+yadeDataDir+'.pkl', 'wb'))
