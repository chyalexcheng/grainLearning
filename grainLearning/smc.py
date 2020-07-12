""" Sequential Monte Carlo: the core of the calibration toolbox
    (https://en.wikipedia.org/wiki/Particle_filter)
"""

import sys, os, glob
import numpy as np
import pickle
from tools import *
from sklearn import mixture
from itertools import repeat


class smc:
    """
    Base class for sequential Monte Carlo (SMC) filtering
    """

    def __init__(self, sigma, ess, obsWeights,
                 yadeVersion='yade-batch', yadeScript='', yadeDataDir='',
                 obsFileName='', obsCtrl='', simDataKeys='', simName='sim',
                 scaleCovWithMax=True, loadSamples=True, skipDEM=True,
                 standAlone=True):
        """
        :param sigma: float, default=1.0
            Normalized (co)variance coefficient

        :param ess: float, default=0.3
            Effective sample size

        :param obsWeights: ndarray of shape (numObs,)
            Relative weights on observation data e.g., np.ones(numObs)

        :param yadeVersion: string, default='yade-batch'

        :param yadeScript: string
            Name of the python script to run Yade

        :param yadeDataDir: string
            Name of the directory where Yade-DEM data is stored

        :param obsFileName: string
            Name of the file where observation data is stored

        :param obsCtrl: string, default=''
            Name of the data sequence to be neglected in Bayesian filtering,
            e.g., strain in strain-controlled triaxial experiments and simulations

        :param simDataKeys: a list of strings, default=['']
            Name of the data sequence to be neglected in Bayesian filtering,
            e.g., strain in strain-controlled triaxial experiments and simulations

        :param simName: string, default='sim'
            Prefix of a simulation file, e.g., <simName>_<key>_<param0>_<param1>_..._<paramN>.txt

        :param loadSamples: bool, default=True
            Whether to load parameter samples from an existing parameter table
            (set to False in the first iteration to create uniform samples from a halton sequence)

        :param scaleCovWithMax: bool, default=True
            Whether to have a time varying covariance matrix or a constant one

        :param skipDEM: bool, default=True
            Whether to skip DEM simulations

        :param standAlone: bool, default=True
            Whether to use GrainLearning as a stand-alone tool to postprocess DEM data
        """

        # SMC parameters
        self.__sigma = sigma
        self.__ess = ess
        self.obsWeights = obsWeights
        self.numParams = 0
        self.numSamples = 0
        self.smcSamples = []

        # SMC data
        self.ips = None
        self.covs = None
        self.posterior = None
        self.likelihood = None
        self.proposal = None

        # hyper-parameters of Bayesian Gaussian mixture
        self.__maxNumComponents = 0
        self.__priorWeight = 0

        # DEM configurations
        self.yadeVersion = yadeVersion
        self.yadeScript = yadeScript
        self.yadeDataDir = yadeDataDir
        self.yadeData = None
        self.paramsFiles = []
        self.paramNames = []
        self.paramRanges = {}

        # Control parameters for observations and simulations
        self.obsFileName = obsFileName
        self.obsCtrl = obsCtrl
        self.simDataKeys = simDataKeys
        self.simName = simName

        # Load observation data
        self.obsData, self.obsCtrlData, self.numObs, self.numSteps = self.getObsDataFromFile(obsFileName, obsCtrl)
        # Assume all observation/reference data are independent
        self.__obsMatrix = np.identity(self.numObs)

        # Whether to use GrainLearning as a stand-alone tool to postprocess DEM data
        self.standAlone = standAlone
        # Whether to have a time varying covariance matrix or a constant one (proportional to maximum observation data)
        self.scaleCovWithMax = scaleCovWithMax
        # Whether to load parameter samples from an existing parameter table
        self.loadSamples = loadSamples
        # whether to skip DEM simulations
        self.skipDEM = skipDEM

        # if run Yade within GrainLearning in Python, initiate a thread pool and parallel scenes
        if not self.standAlone:
            from collision import createScene, runCollision, addSimData
            self.__pool = get_pool(mpi=False, threads=self.numSamples)
            self.__scenes = self.__pool.map(createScene, range(self.numSamples))

    def initialize(self, paramNames, paramRanges, numSamples, maxNumComponents, priorWeight, paramsFile='',
                   proposalFile='', threads=4):
        """
        :param paramNames: list of size (numParams)
        :param paramRanges: dictionary with paramNames being the key
        :param numSamples: int
        :param maxNumComponents: int, default=numSamples/10
        :param priorWeight: float, default=1./maxNumComponents
            weight_concentration_prior of the BayesianGaussianMixture class
            (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)
        :param paramsFile: string
            Current parameter table in a text file
        :param proposalFile: string
            Proposal density trained in the previous iteration and used to generate samples in paramsFile
        :param threads: int
        """
        self.paramNames = paramNames
        self.numParams = len(self.getNames())

        # initialize parameters for training Variational Gaussian mixture models
        self.__maxNumComponents = maxNumComponents
        self.__priorWeight = priorWeight
        # read in parameters if self.smcSamples is an empty list
        if not self.smcSamples:
            # read in parameters from paramsFile, either generated by a halton sequence or a Gaussian mixture model
            if self.loadSamples:
                self.numSamples, _ = self.getParamsFromTable(paramsFile)
            # initialize parameter samples uniformly for the first iteration
            else:
                self.numSamples = self.getInitParams(paramRanges, numSamples, threads)

        # DEM simulation data
        self.yadeData = np.zeros([self.numSteps, self.numSamples, self.numObs])
        # SMC data (ensemble mean, coefficient of variation, posterior, likelihood, and proposal probability
        self.ips = np.zeros([self.numParams, self.numSteps])
        self.covs = np.zeros([self.numParams, self.numSteps])
        self.posterior = np.zeros([self.numSamples, self.numSteps])
        self.likelihood = np.zeros([self.numSamples, self.numSteps])
        if proposalFile:
            # load proposal density from file
            self.proposal = self.loadProposalFromFile(proposalFile, -1)
        else:
            self.proposal = np.ones([self.numSamples]) / self.numSamples

    def getProposalFromSamples(self, iterNO):
        if not self.getSmcSamples():
            RuntimeError("SMC samples not yet loaded...")
        else:
            gmm = mixture.BayesianGaussianMixture(n_components=self.__maxNumComponents,
                                                  weight_concentration_prior=self.__priorWeight, covariance_type='full',
                                                  tol=1e-5, max_iter=int(1e5), n_init=100)
            gmm.fit(self.getSmcSamples()[iterNO])
            proposal = np.exp(gmm.score_samples(self.getSmcSamples()[iterNO]))
        return proposal / sum(proposal)

    def loadProposalFromFile(self, proposalFile, iterNO):
        if not self.getSmcSamples():
            RuntimeError("SMC samples not yet loaded...")
        else:
            proposalModel = pickle.load(open(proposalFile, 'rb'))
            proposal = np.exp(proposalModel.score_samples(self.getSmcSamples()[iterNO]))
        return proposal / sum(proposal)

    def run(self, iterNO=-1, reverse=False):
        # if iterating, reload observation data
        if iterNO > 0:
            self.obsData, self.obsCtrlData, self.numObs, self.numSteps = \
                self.getObsDataFromFile(self.obsFileName, self.obsCtrl)
        # if use Bayesian filtering as a stand alone tool (data already exist before hand)
        if self.standAlone:
            # if run DEM simulations now, with the new parameter table
            if not self.skipDEM and not self.loadSamples:
                # run DEM simulations in batch.
                raw_input("*** Press confirm if the yade file name is correct... ***\n" + self.yadeScript
                          + "\nAbout to run Yade in batch mode with " +
                          ' '.join([self.yadeVersion, self.paramsFiles[iterNO], self.yadeScript]))
                os.system(' '.join([self.yadeVersion, self.paramsFiles[iterNO], self.yadeScript]))
                print 'All simulations finished'
            # if run DEM simulations outside, with the new parameter table
            elif self.skipDEM and not self.loadSamples:
                print 'Leaving GrainLearning... only the parameter table is generated'
                sys.exit()
            # if process the simulation data obtained with the existing parameter table
            else:
                print 'Skipping DEM simulations, read in pre-existing simulation data now'
            # get simulation data from yadeDataDir
            yadeDataFiles = glob.glob(os.getcwd() + '/' + self.yadeDataDir + '/*' + self.simName + '*')
            yadeDataFiles.sort()
            # if glob.glob(self.yadeDataDir + '/*_*txt*') does not return the list of yade data file
            while len(yadeDataFiles) == 0:
                self.simName = raw_input("No DEM filename can be found, tell me the simulation name...\n ")
                yadeDataFiles = glob.glob(os.getcwd() + '/' + self.yadeDataDir + '/*' + self.simName + '*')
                yadeDataFiles.sort()
            # check if parameter combinations match with the simulation filename.
            for i, f in enumerate(yadeDataFiles):
                # get the file name fore the suffix
                f = f.split('.' + f.split('.')[-1])[0]
                # get parameters from the remaining string
                paramsString = f.split('_')[-self.numParams:]
                # element wise comparison of the parameter vector
                if not (np.equal(np.float64(paramsString), self.getSmcSamples()[-1][i]).all()):
                    raise RuntimeError(
                        "Parameters " + ", ".join(
                            ["%s" % v for v in self.getSmcSamples()[-1][i]]) + " do not match with data file " + f)
            # read simulation data into yadeData and drop keys in obsData
            self.getYadeData(yadeDataFiles)
            # if iteration number is an odd number, reverse the data sequences to ensure continuity
            if reverse:
                self.obsCtrlData = self.obsCtrlData[::-1]
                self.obsData = self.obsData[::-1, :]
                self.yadeData = self.yadeData[::-1, :, :]
            # loop over data assimilation steps
            for i in xrange(self.numSteps):
                self.likelihood[:, i], self.posterior[:, i], \
                self.ips[:, i], self.covs[:, i] = self.recursiveBayesian(i)
            # iterate if effective sample size is too big
            while self.getEffectiveSampleSize()[-1] > self.__ess:
                self.__sigma *= 0.9
                for i in xrange(self.numSteps):
                    self.likelihood[:, i], self.posterior[:, i], \
                    self.ips[:, i], self.covs[:, i] = self.recursiveBayesian(i)
        # if perform Bayesian filtering while DEM simulations are running
        else:
            # parameter list
            paramsList = []
            for i in range(self.numSamples):
                paramsForEach = {}
                for j, name in enumerate(self.paramNames):
                    paramsForEach[name] = self.smcSamples[iterNO][i][j]
                paramsList.append(paramsForEach)
            # pass parameter list to simulation instances FIXME: runCollision is the user-defined Yade script
            simData = self.__pool.map(runCollision, zip(self.__scenes, paramsList, repeat(self.obsCtrlData)))
            self.__pool.close()
            # ~ data = runCollision([self.smc__scenes,paramsList[0]])
            # get observation and simulation data ready for Bayesian filtering
            self.obsData = np.array([self.obsData[name] for name in self.simDataKeys]).transpose()
            for i, data in enumerate(simData):
                for j, name in enumerate(self.simDataKeys):
                    # ~ print len(data[name]),i
                    self.yadeData[:, i, j] = data[name]
            # ~ print np.linalg.norm(data[self.obsCtrl]-self.obsCtrlData)
            # loop over data assimilation steps
            if reverse:
                self.obsCtrlData = self.obsCtrlData[::-1]
                self.obsData = self.obsData[::-1, :]
                self.yadeData = self.yadeData[::-1, :, :]
            for i in xrange(self.numSteps):
                self.likelihood[:, i], self.posterior[:, i], \
                self.ips[:, i], self.covs[:, i] = self.recursiveBayesian(i)
            # iterate if effective sample size is too big
            while self.getEffectiveSampleSize()[-1] > self.__ess:
                self.__sigma *= 0.9
                for i in xrange(self.numSteps):
                    self.likelihood[:, i], self.posterior[:, i], \
                    self.ips[:, i], self.covs[:, i] = self.recursiveBayesian(i)
        return self.ips, self.covs

    def getYadeData(self, yadeDataFiles):
        if 0 in self.yadeData.shape: raise RuntimeError, "number of Observations, samples or steps undefined!"
        for i, f in enumerate(yadeDataFiles):
            # if do not know the data to control simulation
            if self.obsCtrl == '':
                yadeData = np.genfromtxt(f)
                for j in range(self.numObs):
                    self.yadeData[:, i, j] = yadeData
            else:
                yadeData = getKeysAndData(f)
                for j, key in enumerate(self.obsData.keys()):
                    self.yadeData[:, i, j] = yadeData[key]
        # if need to remove the data that controls the simulations
        if self.obsCtrl != '':
            obsData = np.zeros([self.numSteps, self.numObs])
            for j, key in enumerate(self.obsData.keys()):
                obsData[:, j] = self.obsData[key]
            self.obsData = obsData

    def recursiveBayesian(self, caliStep, iterNO=-1):
        likelihood = self.getLikelihood(caliStep)
        posterior = self.update(caliStep, likelihood)
        # get ensemble averages and coefficients of variance
        ips = np.zeros(self.numParams)
        covs = np.zeros(self.numParams)
        for i in xrange(self.numParams):
            # ensemble average
            ips[i] = self.smcSamples[iterNO][:, i].dot(posterior)
            # diagonal variance
            covs[i] = ((self.smcSamples[iterNO][:, i] - ips[i]) ** 2).dot(posterior)
            # get coefficient of variance cov
            covs[i] = np.sqrt(covs[i]) / ips[i]
        return likelihood, posterior, ips, covs

    def getLikelihood(self, caliStep):
        # state vector y_t = H(x_t)+Sigma_t
        stateVec = self.yadeData[caliStep, :, :].dot(self.__obsMatrix)
        obsVec = self.obsData[caliStep, :]
        # row-wise substraction obsVec[numObs]-stateVec[numSamples,numObs]
        vecDiff = obsVec - stateVec
        Sigma = self.getCovMatrix(caliStep, self.obsWeights)
        invSigma = np.linalg.inv(Sigma)
        likelihood = np.zeros(self.numSamples)
        # compute likelihood = exp(-0.5*(y_t-H(x_t))*Sigma_t^{-1}*(y_t-H(x_t)))
        for i in xrange(self.numSamples):
            power = (vecDiff[i, :]).dot(invSigma.dot(vecDiff[i, :].T))
            likelihood[i] = np.exp(-0.5 * power)
        # regularize likelihood
        likelihood /= np.sum(likelihood)
        return likelihood

    def update(self, caliStep, likelihood):
        # update posterior probability according to Bayes' rule
        posterior = np.zeros(self.numSamples)
        if caliStep == 0:
            posterior = likelihood / self.proposal
        else:
            posterior = self.posterior[:, caliStep - 1] * likelihood
        # regularize likelihood
        posterior /= np.sum(posterior)
        return posterior

    def getCovMatrix(self, caliStep, weights):
        Sigma = np.zeros([self.numObs, self.numObs])
        # scale observation data with normalized variance parameter to get covariance matrix
        for i in xrange(self.numObs):
            # give smaller weights for better agreement
            if self.scaleCovWithMax:
                Sigma[i, i] = self.__sigma * weights[i] * max(self.obsData[:, i]) ** 2
            else:
                Sigma[i, i] = self.__sigma * weights[i] * self.obsData[caliStep, i] ** 2
        return Sigma

    def getInitParams(self, paramRanges, numSamples, threads):
        """
        Generate the initial parameter sample using a halton sequence
        :param paramRanges: dictionary with paramNames being the ke
        :param numSamples: int
        :param threads: int
        :return: number of parameter samples
        """
        if not paramRanges:
            raise RuntimeError(
                "Parameter range not given. Define the dictionary-type paramRanges or set loadSamples True")
        self.paramRanges = paramRanges
        names = self.getNames()
        minsAndMaxs = np.array([paramRanges[key] for key in names])
        mins = minsAndMaxs[:, 0]
        maxs = minsAndMaxs[:, 1]
        print('Parameters to be identified:', ", ".join(names), '\nMins:', mins, '\nMaxs:', maxs)
        initSmcSamples, initparamsFile = initParamsTable(keys=names, maxs=maxs, mins=mins, num=numSamples,
                                                         threads=threads)
        self.smcSamples.append(np.array(initSmcSamples))
        self.paramsFiles.append(initparamsFile)
        return numSamples

    def getParamsFromTable(self, paramsFile, iterNO=-1):
        """
        :param paramsFile: string
            Current parameter table in a text file
        :param iterNO: int, default=-1
            Index of self.smcSamples
        :return: number of parameter samples and number of unknown parameters
        """

        # if a paramsFile exist, read in parameters directly
        if os.path.exists(paramsFile):
            self.paramsFiles.append(paramsFile)
            # TODO The following assumes unknown parameters in the last self.numParams columns. How to generalize?
            smcSamples = np.genfromtxt(paramsFile, comments='!')[:, -self.numParams:]
            # TODO Still check whether samples read from the paramsFile and simData fileNames are the same

        # if cannot find a paramsFile, extract parameter values from the names of simulation data files
        else:
            print('Cannot find your paramsFile. Will try to get them from the names of simData files...')
            yadeDataFiles = glob.glob(os.getcwd() + '/' + self.yadeDataDir + '/*' + self.simName + '*')
            yadeDataFiles.sort()
            # if self.simName is incorrect and thus cannot get the simulation data files
            while len(yadeDataFiles) == 0:
                self.simName = raw_input("Simulation files cannot be found. Please give the correct simName...\n ")
                yadeDataFiles = glob.glob(os.getcwd() + '/' + self.yadeDataDir + '/*' + self.simName + '*')
                yadeDataFiles.sort()
            # initialize the current smcSamples
            smcSamples = np.zeros([len(yadeDataFiles), self.numParams])
            for i, f in enumerate(yadeDataFiles):
                # get the file type (e.g., txt)
                suffix = '.' + f.split('.')[-1]
                # get the string before the file type
                f = f.split(suffix)[0]
                # get parameters from the remaining string
                # TODO The following assumes the last self.numParams values is the parameter sample. How to generalize?
                paramsString = f.split('_')[-self.numParams:]
                # if the number of strings after <simName> and <key> is different from self.numParams, throw an error
                if len(f.split('_')[2:]) != self.numParams:
                    RuntimeError(
                        'Number of parameters extracted from the file name is different from self.numParams\n\
                         Check if your simulation data file is named as <simName>_<key>_<param0>_<param1>_..._<paramN>.txt')
                # convert strings to float numbers
                smcSamples[i, :] = np.float64(paramsString)
        # append parameter samples to self.smcSamples
        self.smcSamples.append(smcSamples)
        return self.smcSamples[iterNO].shape

    def getObsDataFromFile(self, obsFileName, obsCtrl):
        # if do not know the data to control simulation
        if self.obsCtrl == '':
            keysAndData = np.genfromtxt(obsFileName)
            # if only one observation data vector exist, reshape it with [numSteps,1]
            if len(keysAndData.shape) == 1:
                keysAndData = keysAndData.reshape([keysAndData.shape[0], 1])
            return keysAndData, None, keysAndData.shape[1], keysAndData.shape[0]
        else:
            keysAndData = getKeysAndData(obsFileName)
            # separate obsCtrl for controlling simulations from obsData
            obsCtrlData = keysAndData.pop(obsCtrl)
            numSteps = len(obsCtrlData)
            numObs = len(keysAndData.keys())
            return keysAndData, obsCtrlData, numObs, numSteps

    def resampleParams(self, caliStep, thread=4, iterNO=-1):
        names = self.getNames()
        smcSamples = self.smcSamples[iterNO]
        numSamples = self.numSamples
        # posterior at caliStep is used as the proposal distribution
        proposal = self.posterior[:, caliStep]
        newSmcSamples, newparamsFile, gmm, maxNumComponents = \
            resampledParamsTable(keys=names, smcSamples=smcSamples, proposal=proposal, num=numSamples, thread=thread,
                                 maxNumComponents=self.__maxNumComponents, priorWeight=self.__priorWeight)
        self.smcSamples.append(newSmcSamples)
        self.paramsFiles.append(newparamsFile)
        return gmm, maxNumComponents

    def getPosterior(self):
        return self.posterior

    def getSmcSamples(self):
        return self.smcSamples

    def getNumSteps(self):
        return self.numSteps

    def getEffectiveSampleSize(self):
        nEff = 1. / sum(self.getPosterior() ** 2)
        return nEff / self.numSamples

    def getNames(self):
        return self.paramNames

    def getObsData(self):
        return np.hstack((self.obsCtrlData.reshape(self.numSteps, 1), self.obsData))

    def trainGMMinTime(self, maxNumComponents, iterNO=-1):
        gmmList = []
        smcSamples = self.smcSamples[iterNO]
        for i in xrange(self.numSteps):
            print 'Train DP mixture at time %i...' % i
            posterior = self.posterior[:, i]
            gmmList.append(getGMMFromPosterior(smcSamples, posterior, maxNumComponents))
        return gmmList

    def removeDegeneracy(self, caliStep=-1):
        effIDs = np.where(self.posterior[:, caliStep] < 0.5)[0]
        self.proposal = self.proposal[effIDs, :]
        self.likelihood = self.likelihood[effIDs, :]
        self.posterior = self.posterior[effIDs, :]
        self.smcSamples[0] = self.smcSamples[0][effIDs, :]
        self.yadeData = self.yadeData[:, effIDs, :]
        self.numSamples = len(effIDs)
        for i in xrange(self.numSteps):
            self.likelihood[:, i], self.posterior[:, i], \
            self.ips[:, i], self.covs[:, i] = self.recursiveBayesian(i, self.proposal[:, i])

    def writeBayeStatsToFile(self, reverse):
        np.savetxt(self.yadeDataDir + '/particle.txt', self.getSmcSamples()[0])
        np.savetxt(self.yadeDataDir + '/IP.txt', self.ips[:, ::(-1) ** reverse].T)
        np.savetxt(self.yadeDataDir + '/weight.txt', self.getPosterior()[:, ::(-1) ** reverse])
