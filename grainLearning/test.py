from smc import *
from plotResults import *
from sciPlots import *
import pickle

# user-defined parameters
inputParams = {}
inputParams['iterPF0'] = [0.44000,[1,1,0.01]]
inputParams['iterPF1'] = [0.07480,[1,1,0.01]]
inputParams['iterPF2'] = [0.02206,[1,1,0.01]]
inputParams['iterPF3'] = [0.00174,[1,1,0.02]]

# normalized variance parameter
yadeFile = 'mcTriax_e.py'
iterNO = int(raw_input("Skip DEM simulations for demonstration. Which iteration to look at?\niterNO (e.g., 0, 1, 2, 3): "))
yadeDataDir = 'iterPF%i'%iterNO
sigma = inputParams[yadeDataDir][0]; ess = 1.0
obsWeights = inputParams[yadeDataDir][1]
obsDataFile = 'obsdata.dat'
obsCtrl = 'e_a'

# ranges of parameters (E, \mu, kr, \mu_r)
paramNames = ['E', 'mu', 'k_r','mu_r']
paramRanges = {'E':[100e9,200e9],'mu':[0.3,0.5],'k_r':[0,1e4],'mu_r':[0.1,0.5]}
numSamples = 100; maxNumComponents = int(numSamples/10); priorWeight = 1e-2
sampleDataFile = 'smcTablePF%i.txt'%iterNO
proposalFile = 'gmm_'+yadeDataDir[:-1]+'%i.pkl'%(iterNO-1) if iterNO != 0 else ''
reverse = True if iterNO%2==1 else False

# initialize the problem
smcTest = smc(sigma,obsWeights,yadeFile,yadeDataDir,obsDataFile,obsCtrl)
smcTest.initialize(paramNames,paramRanges,numSamples,maxNumComponents,priorWeight,sampleDataFile=sampleDataFile,loadSamples=True,proposalFile=proposalFile)
# run sequential Monte Carlo; return means and coefficients of variance of PDF over the parameters
ips, covs = smcTest.run(skipDEM=True,reverse=reverse)
# get the parameter samples (ensemble) and posterior probability
posterior = smcTest.getPosterior()
smcSamples = smcTest.getSmcSamples()
# calculate effective sample size
ess = smcTest.getEffectiveSampleSize()[-1]
print 'Effective sample size: %f'%ess

# plot time evolution of effective sample size
plt.figure();plt.plot(smcTest.getEffectiveSampleSize());plt.xlabel('time');plt.ylabel('Effective sample size');
plt.figure();plt.plot(smcTest.getSmcSamples()[0][:,0],smcTest._proposal,'o')
plt.xlabel(paramNames[0]);plt.ylabel('Proposal density');plt.show()

# plot means of PDF over the parameters
microParamUQ = plotIPs(paramNames,ips[:,::(-1)**reverse].T,covs[:,::(-1)**reverse].T,smcTest.getNumSteps(),posterior,smcSamples[0])

# resample parameters
caliStep = -1
gmm, maxNumComponents = smcTest.resampleParams(caliStep=caliStep)

# plot initial and resampled parameters
plotAllSamples(smcTest.getSmcSamples(),smcTest.getNames())

#~ # save trained Gaussian mixture model
#~ pickle.dump(gmm, open(yadeDataDir+'/gmm_'+yadeDataDir+'.pkl', 'wb'))

# get top three realizations with high probabilities
m = smcTest.getNumSteps(); n = smcTest._numSamples
weights = smcTest.getPosterior()*np.repeat(smcTest._proposal,m).reshape(n,m)
weights /= sum(weights); mcFiles = glob.glob(yadeDataDir+'/*txt'); mcFiles.sort()
goodFiles = []; EValues = []; muValues = []; krValues = []; mu_rValues = []
for i in (-weights[:,caliStep]).argsort()[:3]:
	goodFiles.append(mcFiles[i]); 
	EValues.append(smcSamples[0][i,0]); muValues.append(smcSamples[0][i,1])
	krValues.append(smcSamples[0][i,2]); mu_rValues.append(smcSamples[0][i,3])

# plot ensemble prediction and realizations that have high probabilities
keysAndData, obsCtrlData, _, _ = smcTest.getObsDataFromFile(obsDataFile,obsCtrl)
macroParamUQ = plotExpAndNum('VAE3',paramNames,'%i'%iterNO,smcTest.getPosterior()[:,::(-1)**reverse],mcFiles,goodFiles,\
	EValues,muValues,krValues,mu_rValues,\
	keysAndData['p'],keysAndData['q'],keysAndData['n'],obsCtrlData*100,np.zeros(smcTest.getNumSteps()))
