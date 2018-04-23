from smc import *
from plotResults import *

# normalized variance parameter
sigma = 0.5
obsWeights = [1,1,0.01]
yadeFile = 'mcTriax_e.py'
yadeDataDir = 'iterPF0'
obsDataFile = 'obsdata.dat'
obsCtrl = 'e_a'

# ranges of parameters (E, \mu, kr, \mu_r)
paraNames = ['E', '\mu', 'k_r','\mu_r']
paramRanges = {'E':[100e9,200e9],'\mu':[0.3,0.5],'k_r':[0,1e4],'\mu_r':[0.1,0.5]}
numSamples = 100
sampleDataFile = 'smcTable'+yadeDataDir[-1]+'.txt'
smcTest = smc(sigma, obsWeights, yadeFile, yadeDataDir, obsDataFile, obsCtrl)
smcTest.initialize(paramRanges, numSamples, sampleDataFile=sampleDataFile, loadSamples=True)

# run sequential Monte Carlo; return means and coefficients of variance of PDF over the parameters
ips, covs = smcTest.run(skipDEM=True)

# get the parameter samples (ensemble) and posterior probability
posterior = smcTest.getPosterior()
smcSamples = smcTest.getSmcSamples()

# plot means of PDF over the parameters
_ = plotIPs(paraNames,ips.T,covs.T,smcTest.getNumSteps(),posterior,smcSamples[-1])

# resample parameters
caliStep = -1
maxNumComponents = 10
smcTest.resampleParams(caliStep=caliStep,maxNumComponents=maxNumComponents)

# plot initial and resampled parameters
plotAllSamples(smcTest.getSmcSamples(),paraNames)
