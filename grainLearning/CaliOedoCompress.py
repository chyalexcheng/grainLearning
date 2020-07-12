""" Author: Hongyang Cheng <chyalexcheng@gmail.com>
    Bayesian calibration of four DEM parameters for DEM simulation of oedometric compression
    (see the paper [1] for further details)
    [1] https://www.sciencedirect.com/science/article/pii/S0045782519300520
"""

# load GrainLearning modulues
from smc import *
from plotResults import *
from sciPlots import *
import pickle

# user-defined parameter: normalized covariance \sigma and weights on three vectors of observation data
inputParams = {'iterPF0': [0.44000, [1, 1, 0.01]],
               'iterPF1': [0.07480, [1, 1, 0.01]],
               'iterPF2': [0.02206, [1, 1, 0.01]],
               'iterPF3': [0.00174, [1, 1, 0.02]]}

# driver code for triaxial compression
yadeFile = 'mcTriax_e.py'
# use pre-run simulation data for calibration
iterNO = int(raw_input("Skip DEM simulations for demonstration. \
    Which iteration to look at?\niterNO (e.g., 0, 1, 2, 3): "))
yadeDataDir = 'iterPF%i' % iterNO

# give ranges of parameter values (E, \mu, kr, \mu_r)
paramNames = ['E', 'mu', 'k_r', 'mu_r']
# use uniform sampling within certin window if we are at the first iteration
paramRanges = {'E': [100e9, 200e9], 'mu': [0.3, 0.5], 'k_r': [0, 1e4], 'mu_r': [0.1, 0.5]}

# choose observation data file
obsDataFile = 'obsdata.dat'
# select data sequence for simulation control
obsCtrl = 'e_a'

# choose normalized covariance and initialized effective sample size as one
sigma, obsWeights = inputParams[yadeDataDir]
ess = 0.2

# define number of samples and maximum Gaussian components
numSamples = 100
maxNumComponents = int(numSamples / 10)
priorWeight = 1e-2

# select the parameter table used for the pre-run simulations
paramsFile = 'smcTablePF%i.txt' % iterNO
# get pre-trained the proposal density
proposalFile = 'gmm_' + yadeDataDir[:-1] + '%i.pkl' % (iterNO - 1) if iterNO != 0 else ''
# reverse data sequence to ensure continuity
reverse = True if iterNO % 2 == 1 else False

# initialize the problem (if loadSamples is False, the script generates an initial parameter table.
# Otherwise process the simulation data, either pre-run or run within python, to generate a new parameter table)
smcTest = smc(sigma, ess, obsWeights, yadeScript=yadeFile, yadeDataDir=yadeDataDir, obsFileName=obsDataFile, obsCtrl=obsCtrl, scaleCovWithMax=False,
              loadSamples=True, skipDEM=True, standAlone=True)
smcTest.initialize(paramNames, paramRanges, numSamples, maxNumComponents, priorWeight, paramsFile=paramsFile,
                   proposalFile=proposalFile)

# run sequential Monte Carlo; return means and coefficients of variance of PDF over the parameters
ips, covs = smcTest.run(reverse=reverse)
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
microParamUQ = plotIPs(paramNames, ips[:, ::(-1) ** reverse].T, covs[:, ::(-1) ** reverse].T, smcTest.getNumSteps(),
                       posterior, smcSamples[-1])

# resample parameters
caliStep = -1
gmm, maxNumComponents = smcTest.resampleParams(caliStep=caliStep)

# plot initial and resampled parameters
plotAllSamples(smcTest.getSmcSamples(), smcTest.getNames())

# save trained Gaussian mixture model
pickle.dump(gmm, open(yadeDataDir + '/gmm_' + yadeDataDir + '.pkl', 'wb'))

# get top three realizations with high probabilities
m = smcTest.getNumSteps();
n = smcTest.numSamples
weights = smcTest.getPosterior() * np.repeat(smcTest.proposal, m).reshape(n, m)
weights /= sum(weights);
mcFiles = glob.glob(yadeDataDir + '/*_*_*_*txt');
mcFiles.sort()
goodFiles = [];
EValues = [];
muValues = [];
krValues = [];
mu_rValues = []
for i in (-weights[:, caliStep]).argsort()[:3]:
    goodFiles.append(mcFiles[i])
    EValues.append(smcSamples[0][i, 0]);
    muValues.append(smcSamples[0][i, 1])
    krValues.append(smcSamples[0][i, 2]);
    mu_rValues.append(smcSamples[0][i, 3])

# plot ensemble prediction and realizations that have high probabilities
keysAndData, obsCtrlData, _, _ = smcTest.getObsDataFromFile(obsDataFile, obsCtrl)
macroParamUQ = plotExpAndNum('VAE3', paramNames, '%i' % iterNO, smcTest.getPosterior()[:, ::(-1) ** reverse], mcFiles,
                             goodFiles, \
                             EValues, muValues, krValues, mu_rValues, \
                             keysAndData['p'], keysAndData['q'], keysAndData['n'], obsCtrlData * 100,
                             np.zeros(smcTest.getNumSteps()))

# ~ turns = [1,17,30,56,80,-1]
# ~ microMacroWeights = []
# ~ for i in turns:
# ~ microMacroWeights.append(microMacroPDF('VAE3', i, smcTest.getSmcSamples()[0].T, smcTest._yadeDataDir, smcTest.getPosterior()[:,::(-1)**reverse], mcFiles, loadWeights=True))
