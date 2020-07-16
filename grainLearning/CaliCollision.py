"""
Author: Hongyang Cheng <chyalexcheng@gmail.com>
Bayesian calibration of Hertzian contact parameters using two-particle collision simulation
"""

import numpy as np

# load GrainLearning modules
from smc import *
from plotResults import *
import pickle
import matplotlib.pylab as plt

# user-defined parameter: normalized covariance
sigma = float(input("Initialize the normalized covariance as : "))
# target effective sample size
ess = 0.3
obsWeights = [1.0]
# maximum number of iterations
maxNumOfIters = 10

# get observation data file (synthetic data from DEM)
ObsData = np.loadtxt('collision.dat')
# add Gaussian noise
noise = np.random.normal(0, 0.1 * max(ObsData[1]), len(ObsData[1]))

# give ranges of parameter values (E, \nu)
paramNames = ['E', 'nu']
numParams = len(paramNames)
# use uniform sampling for the first iteration
paramRanges = {'E': [10e9, 100e9], 'nu': [0.1, 0.5]}
# key for simulation control
obsCtrl = 'u'
# key for output data
simDataKeys = ['f']

# set number of samples per iteration (e.g., num1D * N * logN for quasi-Sequential Monte Carlo)
numSamples1D = 10
numSamples = int(numSamples1D * numParams * log(numParams))
# set the maximum Gaussian components and prior weight
maxNumComponents = int(numSamples / 10)
priorWeight = 1. / maxNumComponents
covType = 'tied'

# write synthetic observation data to file
obsDataFile = open('collisionObs.dat', 'w')
obsDataFile.write('#\t\tu\t\tf\n')
for i in range(len(ObsData[1])):
    obsDataFile.write('%s\t\t%s\n' % (ObsData[0][i], noise[i] + ObsData[1][i]))
obsDataFile.close()

# initialize the problem
smcTest = smc(sigma, ess, obsWeights, obsCtrl=obsCtrl, simDataKeys=simDataKeys, obsFileName='collisionObs.dat',
              loadSamples=False, runYadeInGL=True, standAlone=False)

# run sequential Monte Carlo; return means and coefficients of variance of PDF over the parameters
iterNO = 0
# iterate the problem
while smcTest.sigma > 1.0e-2 and iterNO < maxNumOfIters:
    # reinitialize the weights
    smcTest.initialize(paramNames, paramRanges, numSamples, maxNumComponents, priorWeight, covType=covType)

    # rerun sequential Monte Carlo
    ips, covs = smcTest.run(iterNO=iterNO, reverse=iterNO % 2)
    # get the parameter samples (ensemble) and posterior probability
    posterior = smcTest.getPosterior()
    smcSamples = smcTest.getSmcSamples()

    # plot means of PDF over the parameters
    plotIPs(paramNames, ips.T, covs.T, smcTest.getNumSteps(), posterior, smcSamples[0])

    # resample parameters
    caliStep = -1
    gmm, maxNumComponents = smcTest.resampleParams(caliStep=caliStep, iterNO=iterNO)

    # plot initial and resampled parameters
    plotAllSamples(smcTest.getSmcSamples(), smcTest.getNames())

    # save trained Gaussian mixture model
    pickle.dump(gmm, open('gmmForCollision_%i' % (iterNO + 1) + '.pkl', 'wb'))

    # plot ground truth and added Gaussian noise
    plt.plot(smcTest.getObsData()[:, 0], smcTest.getObsData()[:, 1], color='grey', label='Ground truth + noise')
    plt.plot(ObsData[0, :], ObsData[1, :], 'ko', markevery=smcTest.numObs/10, label='Ground truth')

    # get top three realizations with high probabilities
    m = smcTest.getNumSteps()
    n = smcTest.numSamples
    weights = smcTest.getPosterior() * np.repeat(smcTest.proposal, m).reshape(n, m)
    weights /= sum(weights)
    for i in (-weights[:, caliStep]).argsort()[:3]:
        plt.plot(smcTest.getObsData()[:, 0], smcTest.yadeData[:, i, 0], label='sim%i' % i)
    plt.xlabel = 'Overlap'; plt.ylabel = 'Force'
    plt.legend(); plt.show()
    # increment iteration NO.
    iterNO += 1
