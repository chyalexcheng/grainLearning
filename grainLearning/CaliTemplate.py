""" Author: Hongyang Cheng <chyalexcheng@gmail.com>
    Bayesian calibration of four DEM parameters for DEM simulation of oedometric compression
    (see the paper [1] for further details)
    [1] https://www.sciencedirect.com/science/article/pii/S0045782519300520
"""

import numpy as np

# load GrainLearning modules
from smc import *
from plotResults import *
from sciPlots import *
import pickle
import matplotlib.pylab as plt

# user-defined parameter: normalized covariance
sigma = float(raw_input("Initialize the normalized covariance as : "))
# target effective sample size
ess = 0.3
obsWeights = [1.0]

yadeDataDir = 'FileBayesIteration0/'
# get observation data file (synthetic data from DEM simulation)
obsDataFile = yadeDataDir + 'Target_Stress.txt'
simName = 'BayesTest'

# give ranges of parameter values ('psd', 'k_n', 'k_t', 'mu')
paramNames = ['psd', 'k_n', 'k_t', 'mu']
# use uniform sampling within certin window if we are at the first iteration
paramRanges = {'psd': [1.2, 10], 'k_n': [1e2, 5e3], 'k_t': [0, 1], 'mu': [0, 0.3]}

# set number of samples per iteration
numSamples = 81
# set the maximum Gaussian components and prior weight
maxNumComponents = int(numSamples / 10)
priorWeight = 1. / maxNumComponents

# select the parameter table used for the pre-run simulations
paramsFile = 'smcTable.txt'

# initialize the problem
smcTest = smc(sigma, ess, obsWeights, yadeDataDir=yadeDataDir, obsDataFile=obsDataFile, simName=simName,
              scaleWithMax=True, loadSamples=True, skipDEM=True, standAlone=True)

smcTest.initialize(paramNames, paramRanges, numSamples, maxNumComponents, priorWeight)

# run sequential Monte Carlo; return means and coefficients of variance of PDF over the parameters
iterNO = 0
ips, covs = smcTest.run(iterNO=iterNO)

# get the parameter samples (ensemble) and posterior probability
posterior = smcTest.getPosterior()
smcSamples = smcTest.getSmcSamples()
# calculate effective sample size
ess = smcTest.getEffectiveSampleSize()[-1]
print 'Effective sample size: %f' % ess

# plot time evolution of effective sample size
plt.figure();
plt.plot(smcTest.getEffectiveSampleSize())
plt.xlabel('time');
plt.ylabel('Effective sample size');
plt.show()

# plot means of PDF over the parameters
microParamUQ = plotIPs(paramNames, ips.T, covs.T, smcTest.getNumSteps(), posterior, smcSamples[0])

# resample parameters
caliStep = -1
gmm, maxNumComponents = smcTest.resampleParams(caliStep=caliStep, iterNO=iterNO)

# plot initial and resampled parameters
plotAllSamples(smcTest.getSmcSamples(), smcTest.getNames())

# get top three realizations with high probabilities
m = smcTest.getNumSteps()
n = smcTest.numSamples
weights = smcTest.getPosterior() * np.repeat(smcTest.proposal, m).reshape(n, m)
weights /= sum(weights)
obsData = smcTest.obsData
plt.plot(np.genfromtxt(yadeDataDir + 'Void_Index.txt'), obsData, label='obs')
for i in (-weights[:, caliStep]).argsort()[:10]:
    plt.plot(np.genfromtxt(yadeDataDir + 'Void_Index.txt'), smcTest.yadeData[:, i, 0], label='sim%i' % i)
plt.legend()
plt.show()