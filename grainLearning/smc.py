""" Sequential Monte Carlo: the core of the calibration toolbox
    (https://en.wikipedia.org/wiki/Particle_filter)
"""

import sys, os, glob
import numpy as np
import pickle
from tools import *
from sklearn import mixture
from collision import createScene, runCollision, addSimData
from itertools import repeat


class smc:
    """
    SMC base class: sequential Monte Carlo (SMC) filtering
    """

    def __init__(self, sigma, ess, obsWeights, yadeFile='', yadeDataDir='', obsDataFile='', obsCtrl='', simDataNames='',
                 simName = 'sim', yadeVersion='yade-batch', standAlone=True):
        # simulation file name (_.py)
        self._yadeVersion = yadeVersion
        self._yadeFile = yadeFile
        self._obsDataFile = obsDataFile
        self._yadeDataDir = yadeDataDir
        self._numParams = 0
        self._numSamples = 0
        self._numObs = 0
        self._numSteps = 0
        self._loadSamples = False
        # normalized variance parameter
        self._sigma = sigma
        self._ess = ess
        self._obsWeights = obsWeights
        self._obsCtrl = obsCtrl
        self._simDataNames = simDataNames
        self._simName = simName
        self._obsData, self._obsCtrlData, self._numObs, self._numSteps = self.getObsDataFromFile(obsDataFile, obsCtrl)
        # assume all measurements are independent
        self._obsMatrix = np.identity(self._numObs)
        self._paramsFiles = []
        self._paramNames = []
        self._paramRanges = {}
        self._smcSamples = []
        # a flag that defines whether Yade is called within python
        self._standAlone = standAlone
        # if run Bayesian filtering on the fly
        if not self._standAlone:
            self.__pool = None
            self.__scenes = None
        # hyper-parameters of Bayesian Gaussian mixture
        self._maxNumComponents = 0
        self._priorWeight = 0
        self._scaleWithMax = 0
        # simulation data
        self._yadeData = None
        self._ips = None
        self._covs = None
        self._posterior = None
        self._likelihood = None
        self._proposal = None

    def initialize(self, paramNames, paramRanges, numSamples, maxNumComponents, priorWeight, paramsFile='',
                   threads=4, loadSamples=False, proposalFile='', scaleWithMax=False):
        # assign parameter names
        self._paramNames = paramNames
        # initialize parameters for Gaussian mixture model
        self._maxNumComponents = maxNumComponents
        self._priorWeight = priorWeight
        # whether change covariance matrix in time or set it constant (proportional to the maximum observation data)
        self._scaleWithMax = scaleWithMax
        # initialize sample uniformly if no parameter table available
        if len(self._smcSamples) == 0:
            if loadSamples:
                self._loadSamples = loadSamples
                self._numSamples, _ = self.getParamsFromTable(paramsFile, paramNames, paramRanges)
            else:
                self._loadSamples = loadSamples
                self._numSamples = self.getInitParams(paramRanges, numSamples, threads)
        # simulation data
        self._yadeData = np.zeros([self._numSteps, self._numSamples, self._numObs])
        # identified optimal parameters
        self._ips = np.zeros([self._numParams, self._numSteps])
        self._covs = np.zeros([self._numParams, self._numSteps])
        self._posterior = np.zeros([self._numSamples, self._numSteps])
        self._likelihood = np.zeros([self._numSamples, self._numSteps])
        self._proposal = np.ones([self._numSamples]) / self._numSamples
        if proposalFile != '':
            # load proposal density from file
            self._proposal = self.loadProposalFromFile(proposalFile, -1)
        # if run Bayesian filtering on the fly
        if not self._standAlone:
            self.__pool = get_pool(mpi=False, threads=self._numSamples)
            self.__scenes = self.__pool.map(createScene, range(self._numSamples))

    # ~ self.__scenes = createScene(0)

    def getProposalFromSamples(self, iterNO):
        if len(self.getSmcSamples()) == 0:
            RuntimeError, "SMC samples not yet loaded..."
        else:
            gmm = mixture.BayesianGaussianMixture(n_components=self._maxNumComponents,
                                                  weight_concentration_prior=self._priorWeight, covariance_type='full',
                                                  tol=1e-5, max_iter=int(1e5), n_init=100)
            gmm.fit(self.getSmcSamples()[iterNO])
            proposal = np.exp(gmm.score_samples(self.getSmcSamples()[iterNO]))
        return proposal / sum(proposal)

    def loadProposalFromFile(self, proposalFile, iterNO):
        if len(self.getSmcSamples()) == 0:
            RuntimeError, "SMC samples not yet loaded..."
        else:
            proposalModel = pickle.load(open(proposalFile, 'rb'))
            proposal = np.exp(proposalModel.score_samples(self.getSmcSamples()[iterNO]))
        return proposal / sum(proposal)

    def run(self, skipDEM=False, iterNO=-1, reverse=False, threads=1):
        # if iterating, reload observation data
        if iterNO > 0:
            self._obsData, self._obsCtrlData, self._numObs, self._numSteps = \
                self.getObsDataFromFile(self._obsDataFile, self._obsCtrl)
        # if use Bayesian filtering as a stand alone tool (data already exist before hand)
        if self._standAlone:
            # if run DEM simulations now, with the new parameter table
            if not skipDEM and not self._loadSamples:
                # run DEM simulations in batch.
                raw_input("*** Press confirm if the yade file name is correct... ***\n" + self._yadeFile
                          + "\nAbout to run Yade in batch mode with " +
                          ' '.join([self._yadeVersion, self._paramsFiles[iterNO], self._yadeFile]))
                os.system(' '.join([self._yadeVersion, self._paramsFiles[iterNO], self._yadeFile]))
                print 'All simulations finished'
            # if run DEM simulations outside, with the new parameter table
            elif skipDEM and not self._loadSamples:
                sys.exit()
            # if process the simulation data obtained with the existing parameter table
            else:
                print 'Skipping DEM simulations, read in data now'
            # get simulation data from yadeDataDir
            yadeDataFiles = glob.glob(os.getcwd()+'/'+self._yadeDataDir + '/*' + self._simName + '*')
            yadeDataFiles.sort()
            # if glob.glob(self._yadeDataDir + '/*_*txt*') does not return the list of yade data file
            while len(yadeDataFiles) == 0:
                self._simName = raw_input("No DEM filename can be found, tell me the simulation name...\n ")
                yadeDataFiles = glob.glob(os.getcwd()+'/'+self._yadeDataDir + '/*' + self._simName + '*')
                yadeDataFiles.sort()
            # read simulation data into yadeData and drop keys in obsData
            self.getYadeData(yadeDataFiles)
            # if iteration number is an odd number, reverse the data sequences to ensure continuity
            if reverse:
                self._obsCtrlData = self._obsCtrlData[::-1]
                self._obsData = self._obsData[::-1, :]
                self._yadeData = self._yadeData[::-1, :, :]
            # loop over data assimilation steps
            for i in xrange(self._numSteps):
                self._likelihood[:, i], self._posterior[:, i], \
                self._ips[:, i], self._covs[:, i] = self.recursiveBayesian(i)
            # iterate if effective sample size is too big
            sigma = self._sigma
            while self.getEffectiveSampleSize()[-1] > self._ess:
                sigma *= 0.9
                for i in xrange(self._numSteps):
                    self._likelihood[:, i], self._posterior[:, i], \
                    self._ips[:, i], self._covs[:, i] = self.recursiveBayesian(i)
        # if perform Bayesian filtering while DEM simulations are running
        else:
            # parameter list
            paramsList = []
            for i in range(self._numSamples):
                paramsForEach = {}
                for j, name in enumerate(self._paramNames):
                    paramsForEach[name] = self._smcSamples[iterNO][i][j]
                paramsList.append(paramsForEach)
            # pass parameter list to simulation instances FIXME: runCollision is the user-defined Yade script
            simData = self.__pool.map(runCollision, zip(self.__scenes, paramsList, repeat(self._obsCtrlData)))
            self.__pool.close()
            # ~ data = runCollision([self._smc__scenes,paramsList[0]])
            # get observation and simulation data ready for Bayesian filtering
            self._obsData = np.array([self._obsData[name] for name in self._simDataNames]).transpose()
            for i, data in enumerate(simData):
                for j, name in enumerate(self._simDataNames):
                    # ~ print len(data[name]),i
                    self._yadeData[:, i, j] = data[name]
            # ~ print np.linalg.norm(data[self._obsCtrl]-self._obsCtrlData)
            # loop over data assimilation steps
            if reverse:
                self._obsCtrlData = self._obsCtrlData[::-1]
                self._obsData = self._obsData[::-1, :]
                self._yadeData = self._yadeData[::-1, :, :]
            for i in xrange(self._numSteps):
                self._likelihood[:, i], self._posterior[:, i], \
                self._ips[:, i], self._covs[:, i] = self.recursiveBayesian(i)
            # iterate if effective sample size is too big
            sigma = self._sigma
            while self.getEffectiveSampleSize()[-1] > self._ess:
                sigma *= 0.9
                for i in xrange(self._numSteps):
                    self._likelihood[:, i], self._posterior[:, i], \
                    self._ips[:, i], self._covs[:, i] = self.recursiveBayesian(i)
        return self._ips, self._covs

    def getYadeData(self, yadeDataFiles):
        if 0 in self._yadeData.shape: raise RuntimeError, "number of Observations, samples or steps undefined!"
        for i, f in enumerate(yadeDataFiles):
            # if do not know the data to control simulation
            if self._obsCtrl == '':
                yadeData = np.genfromtxt(f)
                for j in range(self._numObs):
                    self._yadeData[:, i, j] = yadeData
            else:
                yadeData = getKeysAndData(f)
                for j, key in enumerate(self._obsData.keys()):
                    self._yadeData[:, i, j] = yadeData[key]
        # if need to remove the data that controls the simulations
        if self._obsCtrl != '':
            obsData = np.zeros([self._numSteps, self._numObs])
            for j, key in enumerate(self._obsData.keys()):
                    obsData[:, j] = self._obsData[key]
            self._obsData = obsData

    def recursiveBayesian(self, caliStep, iterNO=-1):
        likelihood = self.getLikelihood(caliStep)
        posterior = self.update(caliStep, likelihood)
        # get ensemble averages and coefficients of variance
        ips = np.zeros(self._numParams)
        covs = np.zeros(self._numParams)
        for i in xrange(self._numParams):
            # ensemble average
            ips[i] = self._smcSamples[iterNO][:, i].dot(posterior)
            # diagonal variance
            covs[i] = ((self._smcSamples[iterNO][:, i] - ips[i]) ** 2).dot(posterior)
            # get coefficient of variance cov
            covs[i] = np.sqrt(covs[i]) / ips[i]
        return likelihood, posterior, ips, covs

    def getLikelihood(self, caliStep):
        # state vector y_t = H(x_t)+Sigma_t
        stateVec = self._yadeData[caliStep, :, :].dot(self._obsMatrix)
        obsVec = self._obsData[caliStep, :]
        # row-wise substraction obsVec[numObs]-stateVec[numSamples,numObs]
        vecDiff = obsVec - stateVec
        Sigma = self.getCovMatrix(caliStep, self._obsWeights)
        invSigma = np.linalg.inv(Sigma)
        likelihood = np.zeros(self._numSamples)
        # compute likelihood = exp(-0.5*(y_t-H(x_t))*Sigma_t^{-1}*(y_t-H(x_t)))
        for i in xrange(self._numSamples):
            power = (vecDiff[i, :]).dot(invSigma.dot(vecDiff[i, :].T))
            likelihood[i] = np.exp(-0.5 * power)
        # regularize likelihood
        likelihood /= np.sum(likelihood)
        return likelihood

    def update(self, caliStep, likelihood):
        # update posterior probability according to Bayes' rule
        posterior = np.zeros(self._numSamples)
        if caliStep == 0:
            posterior = likelihood / self._proposal
        else:
            posterior = self._posterior[:, caliStep - 1] * likelihood
        # regularize likelihood
        posterior /= np.sum(posterior)
        return posterior

    def getCovMatrix(self, caliStep, weights):
        Sigma = np.zeros([self._numObs, self._numObs])
        # scale observation data with normalized variance parameter to get covariance matrix
        for i in xrange(self._numObs):
            # give smaller weights for better agreement
            if self._scaleWithMax:
                Sigma[i, i] = self._sigma * weights[i] * max(self._obsData[:, i]) ** 2
            else:
                Sigma[i, i] = self._sigma * weights[i] * self._obsData[caliStep, i] ** 2
        return Sigma

    def getInitParams(self, paramRanges, numSamples, threads):
        if len(paramRanges.values()) == 0:
            raise RuntimeError(
                "Parameter range is not given. Define the dictionary-type variable paramRanges or loadSamples True")
        self._paramRanges = paramRanges
        self._numParams = len(self.getNames())
        names = self.getNames()
        minsAndMaxs = np.array([paramRanges[key] for key in self.getNames()])
        mins = minsAndMaxs[:, 0]
        maxs = minsAndMaxs[:, 1]
        print 'Parameters to be identified:', ", ".join(names), '\nMins:', mins, '\nMaxs:', maxs
        initSmcSamples, initparamsFile = initParamsTable(keys=names, maxs=maxs, mins=mins, num=numSamples,
                                                         threads=threads)
        self._smcSamples.append(np.array(initSmcSamples))
        self._paramsFiles.append(initparamsFile)
        return numSamples

    def getParamsFromTable(self, paramsFile, names, paramRanges, iterNO=-1):
        self._paramRanges = paramRanges
        self._numParams = len(self.getNames())
        # if paramsFile exist, read in parameters and append data to self._smcSamples
        if os.path.exists(paramsFile):
            self._paramsFiles.append(paramsFile)
            smcSamples = np.genfromtxt(paramsFile, comments='!')[:, -self._numParams:]
            self._smcSamples.append(smcSamples)
            return self._smcSamples[iterNO].shape
        # if paramsFile does not exist, get parameter values from the names of simulation files
        else:
            yadeDataFiles = glob.glob(os.getcwd()+'/'+self._yadeDataDir + '/*' + self._simName + '*')
            yadeDataFiles.sort()
            while len(yadeDataFiles) == 0:
                self._simName = raw_input("No DEM filename can be found, tell me the simulation name...\n ")
                yadeDataFiles = glob.glob(os.getcwd()+'/'+self._yadeDataDir + '/*' + self._simName + '*')
                yadeDataFiles.sort()
            smcSamples = np.zeros([len(yadeDataFiles), len(self.getNames())])
            for i, f in enumerate(yadeDataFiles):
                # FIXME how to make the split of parameters automated. Now this is the parameter names are hard coded
                # f = f.split('.results')[0]
                # _, key, psd, k_n, k_t, mu = f.split('_')
                f = f.split('.txt')[0]
                _, key, E, mu_i, mu, k_r, mu_r = f.split('_')
                smcSamples[i, :] = eval(' '.join(['abs(float(' + name + ')),' for name in names])[:-1])
            self._smcSamples.append(smcSamples)
            return self._smcSamples[iterNO].shape

    def getObsDataFromFile(self, obsDataFile, obsCtrl):
        # if do not know the data to control simulation
        if self._obsCtrl == '':
            keysAndData = np.genfromtxt(obsDataFile)
            # if only one observation data vector exist, reshape it with [numSteps,1]
            if len(keysAndData.shape) == 1:
                keysAndData = keysAndData.reshape([keysAndData.shape[0], 1])
            return keysAndData, None, keysAndData.shape[1], keysAndData.shape[0]
        else:
            keysAndData = getKeysAndData(obsDataFile)
            # separate obsCtrl for controlling simulations from obsData
            obsCtrlData = keysAndData.pop(obsCtrl)
            numSteps = len(obsCtrlData)
            numObs = len(keysAndData.keys())
            return keysAndData, obsCtrlData, numObs, numSteps

    def resampleParams(self, caliStep, thread=4, iterNO=-1):
        names = self.getNames()
        smcSamples = self._smcSamples[iterNO]
        numSamples = self._numSamples
        # posterior at caliStep is used as the proposal distribution
        proposal = self._posterior[:, caliStep]
        newSmcSamples, newparamsFile, gmm, maxNumComponents = \
            resampledParamsTable(keys=names, smcSamples=smcSamples, proposal=proposal, num=numSamples, thread=thread,
                                 maxNumComponents=self._maxNumComponents, priorWeight=self._priorWeight)
        self._smcSamples.append(newSmcSamples)
        self._paramsFiles.append(newparamsFile)
        return gmm, maxNumComponents

    def getPosterior(self):
        return self._posterior

    def getSmcSamples(self):
        return self._smcSamples

    def getNumSteps(self):
        return self._numSteps

    def getEffectiveSampleSize(self):
        nEff = 1. / sum(self.getPosterior() ** 2)
        return nEff / self._numSamples

    def getNames(self):
        return self._paramNames

    def getObsData(self):
        return np.hstack((self._obsCtrlData.reshape(self._numSteps, 1), self._obsData))

    def trainGMMinTime(self, maxNumComponents, iterNO=-1):
        gmmList = []
        smcSamples = self._smcSamples[iterNO]
        for i in xrange(self._numSteps):
            print 'Train DP mixture at time %i...' % i
            posterior = self._posterior[:, i]
            gmmList.append(getGMMFromPosterior(smcSamples, posterior, maxNumComponents))
        return gmmList

    def removeDegeneracy(self, caliStep=-1):
        effIDs = np.where(self._posterior[:, caliStep] < 0.5)[0]
        self._proposal = self._proposal[effIDs, :]
        self._likelihood = self._likelihood[effIDs, :]
        self._posterior = self._posterior[effIDs, :]
        self._smcSamples[0] = self._smcSamples[0][effIDs, :]
        self._yadeData = self._yadeData[:, effIDs, :]
        self._numSamples = len(effIDs)
        for i in xrange(self._numSteps):
            self._likelihood[:, i], self._posterior[:, i], \
            self._ips[:, i], self._covs[:, i] = self.recursiveBayesian(i, self._proposal[:, i])

    def writeBayeStatsToFile(self, reverse):
        np.savetxt(self._yadeDataDir + '/particle.txt', self.getSmcSamples()[0])
        np.savetxt(self._yadeDataDir + '/IP.txt', self._ips[:, ::(-1) ** reverse].T)
        np.savetxt(self._yadeDataDir + '/weight.txt', self.getPosterior()[:, ::(-1) ** reverse])

# ~ turns = [1,17,30,56,80,-1]
# ~ microMacroWeights = []
# ~ for i in turns:
# ~ microMacroWeights.append(microMacroPDF('VAE3', i, smcTest.getSmcSamples()[0].T, smcTest._yadeDataDir, smcTest.getPosterior()[:,::(-1)**reverse], mcFiles, loadWeights=True))
