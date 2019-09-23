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
# number of iterations
numOfIters = 4

# get observation data file (synthetic data from DEM simulation)
obsDataFile = 'collision.dat'
ObsData = np.loadtxt('collision.dat')
# select data sequence for simulation control
obsCtrl = 'u'
simDataKeys = ['f']
# add Gaussian noise
noise = np.random.normal(0, 0.04 * max(ObsData[1]), len(ObsData[1]))

# give ranges of parameter values (E, \mu, kr, \mu_r)
paramNames = ['E', 'nu', 'mu', 'safe']
# use uniform sampling within certin window if we are at the first iteration
paramRanges = {'E': [10e9, 100e9], 'nu': [0.1, 0.5], 'mu': [0, 1.0], 'safe': [0.01, 1.0]}

# set number of samples per iteration
numSamples = 12
# set the maximum Gaussian components and prior weight
maxNumComponents = int(numSamples / 10);
priorWeight = 1. / maxNumComponents

# write synthetic observation data to file
obsDataFile = open('collisionObs.dat', 'w')
obsDataFile.write('#\t\tu\t\tf\n')
for i in range(len(ObsData[1])):
    obsDataFile.write('%s\t\t%s\n' % (ObsData[0][i], noise[i] + ObsData[1][i]))
obsDataFile.close()

# initialize the problem
smcTest = smc(sigma, ess, obsWeights, obsCtrl=obsCtrl, simDataKeys=simDataKeys, obsDataFile='collisionObs.dat',
              loadSamples=False, standAlone=False)
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
plt.figure();
plt.plot(smcTest.getSmcSamples()[0][:, 0], smcTest.proposal, 'o')
plt.xlabel(paramNames[0]);
plt.ylabel('Proposal density');
plt.show()

# plot means of PDF over the parameters
microParamUQ = plotIPs(paramNames, ips.T, covs.T, smcTest.getNumSteps(), posterior, smcSamples[0])

# resample parameters
caliStep = -1
gmm, maxNumComponents = smcTest.resampleParams(caliStep=caliStep, iterNO=iterNO)

# plot initial and resampled parameters
plotAllSamples(smcTest.getSmcSamples(), smcTest.getNames())

# save trained Gaussian mixture model
pickle.dump(gmm, open('gmmForCollision_%i.pkl' % iterNO, 'wb'))

# get top three realizations with high probabilities
m = smcTest.getNumSteps()
n = smcTest.numSamples
weights = smcTest.getPosterior() * np.repeat(smcTest.proposal, m).reshape(n, m)
weights /= sum(weights)
obsData = smcTest.getObsData()
plt.plot(obsData[:, 0], obsData[:, 1], label='obs')
for i in (-weights[:, caliStep]).argsort()[:3]: plt.plot(obsData[:, 0], smcTest.yadeData[:, i, 0], label='sim%i' % i)
plt.legend();
plt.show()

# iterate the problem
for i in range(numOfIters):
    iterNO = i + 1
    # reinitialize the weights
    smcTest.initialize(paramNames, paramRanges, numSamples, maxNumComponents, priorWeight)

    # rerun sequential Monte Carlo
    ips, covs = smcTest.run(iterNO=iterNO)
    # get the parameter samples (ensemble) and posterior probability
    posterior = smcTest.getPosterior()
    smcSamples = smcTest.getSmcSamples()
    # calculate effective sample size
    ess = smcTest.getEffectiveSampleSize()[-1]
    print 'Effective sample size: %f' % ess

    # plot means of PDF over the parameters
    microParamUQ = plotIPs(paramNames, ips.T, covs.T, smcTest.getNumSteps(), posterior, smcSamples[0])
    # resample parameters
    caliStep = -1
    gmm, maxNumComponents = smcTest.resampleParams(caliStep=caliStep, iterNO=iterNO)
    # plot initial and resampled parameters
    plotAllSamples(smcTest.getSmcSamples(), smcTest.getNames())
    # save trained Gaussian mixture model
    pickle.dump(gmm, open('gmmForCollision_%i' % (i + 1) + '.pkl', 'wb'))
    # get top three realizations with high probabilities
    m = smcTest.getNumSteps()
    n = smcTest.numSamples
    weights = smcTest.getPosterior() * np.repeat(smcTest.proposal, m).reshape(n, m)
    weights /= sum(weights)
    obsData = smcTest.getObsData()
    plt.plot(obsData[:, 0], obsData[:, 1], label='obs')
    for i in (-weights[:, caliStep]).argsort()[:3]: plt.plot(obsData[:, 0], smcTest.yadeData[:, i, 0],
                                                             label='sim%i' % i)
    plt.legend();
    plt.show()
