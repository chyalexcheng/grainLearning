from smc import *
from plotResults import *

# normalized variance parameter
sigma = 0.3
obsWeights = [1,1,0.01]
yadeFile = 'mcTriax_e.py'
yadeDataDir = 'iterPF0'
obsDataFile = 'obsdata.dat'
obsCtrl = 'e_a'

# ranges of parameters (E, \mu, kr, \mu_r)
paraNames = ['E', 'mu', 'kr','mu_r']
paramRanges = {'E':[100e9,200e9],'mu':[0.3,0.5],'kr':[0,1e4],'mu_r':[0.1,0.5]}
numSamples = 100
sampleDataFile = 'smcTable.txt'
smcTest = smc(sigma, obsWeights, yadeFile, yadeDataDir, obsDataFile, obsCtrl)
smcTest.initialize(paramRanges, numSamples, sampleDataFile=sampleDataFile, loadSamples=True)
smcTest.run(skipDEM=True)

ips = smcTest._ips
posterior = smcTest._posterior
smcSamples = smcTest._smcSamples
_ = plotIPs(paraNames,ips.T,smcTest._numSteps,posterior,smcSamples)
