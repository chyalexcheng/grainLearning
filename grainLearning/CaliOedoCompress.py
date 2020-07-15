""" Author: Hongyang Cheng <chyalexcheng@gmail.com>
    Bayesian calibration of four DEM parameters for DEM simulation of oedometric compression
    (see the paper [1] for details)
    [1] https://www.sciencedirect.com/science/article/pii/S0045782519300520
"""

# load GrainLearning modules
from smc import *
from plotResults import *
from sciPlots import *
import pickle

# name of the driver script for triaxial compression
yadeFile = 'mcTriax_e.py'
# use pre-run simulation data for calibration
iterNO = int(input("Skip DEM simulations for demonstration. \
    Which iteration to look at?\niterNO (e.g., 0, 1, 2, 3): "))
yadeDataDir = 'iterPF%i' % iterNO

sciPlot = True
writePlots = False

# user-defined parameter
# 1. normalized covariance sigma
# 2. weights on three vectors of observation data
inputParams = {'iterPF0': [0.44000, [1, 1, 0.01]],
               'iterPF1': [0.07480, [1, 1, 0.01]],
               'iterPF2': [0.02206, [1, 1, 0.01]],
               'iterPF3': [0.00174, [1, 1, 0.02]]}
sigma, obsWeights = inputParams[yadeDataDir]

# give ranges of parameter values (E, \mu, kr, \mu_r)
paramNames = ['E', 'mu', 'k_r', 'mu_r']
# use uniform sampling for the first iteration
paramRanges = {'E': [100e9, 200e9], 'mu': [0.3, 0.5], 'k_r': [0, 1e4], 'mu_r': [0.1, 0.5]}

# name of the observation data file
obsDataFile = 'obsdata.dat'
# name of the data sequence that controls simulation
obsCtrl = 'e_a'

# choose an appropriate effective sample size, e.g., 0.2
ess = 0.2

# define number of samples and maximum Gaussian components
numSamples = 100
maxNumComponents = int(numSamples / 10)
priorWeight = 1.0e-2

# name of the parameter table that corresponds to simData in yadeDataDir
paramsFile = 'smcTablePF%i.txt' % iterNO
# pre-trained proposal density in the pickle format
proposalFile = 'gmm_' + yadeDataDir[:-1] + '%i.pkl' % (iterNO - 1) if iterNO != 0 else ''
# if iteration number is an odd number, reverse the data sequences to ensure data continuity
reverse = True if iterNO % 2 == 1 else False

# initialize the Sequential Monte Carlo problem
smcTest = smc(sigma, ess, obsWeights,
              yadeScript=yadeFile, yadeDataDir=yadeDataDir,
              obsFileName=obsDataFile, obsCtrl=obsCtrl, simName='VAE',
              scaleCovWithMax=False, loadSamples=True, runYadeInGL=False, standAlone=True)
smcTest.initialize(paramNames, paramRanges, numSamples, maxNumComponents, priorWeight, paramsFile=paramsFile,
                   proposalFile=proposalFile)

# run sequential Monte Carlo and return ensemble means and coefficients of variance
ips, covs = smcTest.run(reverse=reverse)
# get the effective sample size
ess = smcTest.getEffectiveSampleSize()[-1]
print('Effective sample size: %f' % ess)

# TODO move the plotting stuff into plotResults.py
if sciPlot:
    # plot time evolution of effective sample size
    fig1 = plt.figure()
    plt.plot(smcTest.getEffectiveSampleSize())
    plt.xlabel('Time')
    plt.ylabel('Effective sample size')
    # plot the distribution of proposal density over the first unknown parameter
    fig2 = plt.figure()
    plt.plot(smcTest.getSmcSamples()[0][:, 0], smcTest.proposal, 'o')
    plt.xlabel(paramNames[0])
    plt.ylabel('Proposal density')
    if writePlots:
        fig1.savefig(yadeDataDir+'/ess.png', dpi=300)
        fig2.savefig(yadeDataDir+'/proposal.png', dpi=300)
    else: plt.show()

    # plot the ensemble means as the identified parameters
    microParamUQ = plotIPs(paramNames, ips[:, ::(-1) ** reverse].T, covs[:, ::(-1) ** reverse].T, smcTest.getNumSteps(),
                           smcTest.getPosterior(), smcTest.getSmcSamples()[-1])

# use posterior distribution at the last calibration step for resampling
caliStep = -1
gmm, maxNumComponents = smcTest.resampleParams(caliStep=caliStep)

# plot the initial and resampled parameters
plotAllSamples(smcTest.getSmcSamples(), smcTest.getNames())

# save trained Gaussian mixture model
pickle.dump(gmm, open(yadeDataDir + '/gmm_' + yadeDataDir + '.pkl', 'wb'))

# get top three realizations with high probabilities
m = smcTest.getNumSteps()
n = smcTest.numSamples
weights = smcTest.getPosterior() * np.repeat(smcTest.proposal, m).reshape(n, m)
weights /= sum(weights)
mcFiles = glob.glob(yadeDataDir + '/*_*_*_*txt'); mcFiles.sort()
goodFiles = []
EValues = []; muValues = []; krValues = []; mu_rValues = []
for i in (-weights[:, caliStep]).argsort()[:3]:
    goodFiles.append(mcFiles[i])
    EValues.append(smcTest.smcSamples[0][i, 0]); muValues.append(smcTest.smcSamples[0][i, 1])
    krValues.append(smcTest.smcSamples[0][i, 2]); mu_rValues.append(smcTest.smcSamples[0][i, 3])

# plot ensemble prediction and realizations that have high probabilities
keysAndData, obsCtrlData, _, _ = smcTest.getObsDataFromFile(obsDataFile, obsCtrl)
macroParamUQ = plotExpAndNum('VAE3', paramNames, '%i' % iterNO, smcTest.getPosterior()[:, ::(-1) ** reverse], mcFiles,
                             goodFiles,
                             EValues, muValues, krValues, mu_rValues,
                             keysAndData['p'], keysAndData['q'], keysAndData['n'], obsCtrlData * 100,
                             np.zeros(smcTest.getNumSteps()))

# # plot statistics over the micro-macro joint space
# if sciPlot:
#     turns = [1, 17, 30, 56, 80, -1]
#     microMacroWeights = []
#     for i in turns:
#         microMacroWeights.append(microMacroPDF('VAE3', i, smcTest.getSmcSamples()[0].T, smcTest.yadeDataDir,
#                                                smcTest.getPosterior()[:, ::(-1) ** reverse], mcFiles,
#                                                loadWeights=False))
