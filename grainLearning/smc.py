""" Sequential Monte Carlo: the core of the calibration toolbox
	(https://en.wikipedia.org/wiki/Particle_filter)
"""

import sys
import os
import glob
import numpy as np
import pickle
from tools import * 
from sklearn import mixture

class smc:
	"""
	"""
	def __init__(self, sigma, obsWeights, yadeFile='', yadeDataDir='', obsDataFile='', obsCtrl='', yadeVersion='yade-batch', standAlone=True):
		# simulation file name (_.py)
		self._yadeVersion = yadeVersion
		self._name = yadeFile
		self._yadeDataDir = yadeDataDir
		self._numParams = 0
		self._numSamples = 0
		self._numObs = 0
		self._numSteps = 0
		# normalized variance parameter
		self._sigma = sigma
		self._obsWeights = obsWeights
		self._obsData, self._obsCtrlData, self._numObs, self._numSteps = self.getObsDataFromFile(obsDataFile,obsCtrl)
		# assume all measurements are independent
		self._obsMatrix = np.identity(self._numObs)
		self._sampleDataFiles = ['']
		self._paramNames = []
		self._paramRanges = {}
		self._smcSamples = []
		# a flag that defines whether Yade is called within python
		self._standAlone = standAlone
		
	def initialize(self, paramNames, paramRanges, numSamples, sampleDataFile='', thread=4, loadSamples=False, proposalFile=''):
		self._paramNames = paramNames
		if loadSamples: self.getParamsFromTable(sampleDataFile, paramNames, paramRanges)
		else: self.getInitParams(paramRanges, numSamples, thread)
		if self._standAlone:
			# simulation data
			self._yadeData = np.zeros([self._numSteps, self._numSamples, self._numObs])
			# identified optimal parameters
			self._ips = np.zeros([self._numParams, self._numSteps])
			self._covs = np.zeros([self._numParams, self._numSteps])
			self._posterior = np.zeros([self._numSamples, self._numSteps])
			self._likelihood = np.zeros([self._numSamples, self._numSteps])
			self._proposal = np.ones([self._numSamples, self._numSteps])/self._numSamples
			self._prior = np.ones([self._numSamples])/self._numSamples
			if proposalFile != '':
				self._prior = self.getPriorFromSamples()
				self.loadProposalFromFile(proposalFile)

	def getPriorFromSamples(self,n_components=15):
		if len(self.getSmcSamples()) == 0:
			RuntimeError,"SMC samples not yet loaded..."
		elif len(self.getSmcSamples()) == 2:
			RuntimeError,"SMC samples already resampled..."
		else:
			gmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type='full',tol = 1e-5,max_iter=int(1e5),n_init=100)
			gmm.fit(self.getSmcSamples()[0])
			prior = np.exp(gmm.score_samples(self.getSmcSamples()[0]))
		return prior/sum(prior)

	def loadProposalFromFile(self,proposalFile):
		if len(self.getSmcSamples()) == 0:
			RuntimeError,"SMC samples not yet loaded..."
		elif len(self.getSmcSamples()) == 2:
			RuntimeError,"SMC samples already resampled..."
		else:
			proposalModelList = pickle.load(open(proposalFile, 'rb'))
			for i, proposalModel in enumerate(proposalModelList):
				proposal = np.exp(proposalModel.score_samples(self.getSmcSamples()[0]))
				self._proposal[:,i] = proposal/sum(proposal)

	def run(self,skipDEM=False,iterNO=-1):
		if self._standAlone:
			if not skipDEM:
				# run DEM simulations in batch. 
				raw_input("*** Press Enter if the file name is correct... ***\n"+self._name\
						  +"\n(Add 'key%03d'%key of sampleDataFiles into DEM filenames)" )
				system(' '.join([self._yadeVersion, self._sampleDataFiles[iterNO], self._name]))
				print 'All simulations finished'
			else: print 'Skipping DEM simulations, read in data now'
			yadeDataFiles = glob.glob(self._yadeDataDir+'/*txt*')
			yadeDataFiles.sort()
			while len(yadeDataFiles) == 0:
				key = raw_input("No DEM filename has key, tell me the key...\n ")
				yadeDataFiles = glob.glob(self._yadeDataDir+'/*'+key+'*')
				yadeDataFiles.sort()
			# read simulation data into self._yadeData and drop keys in obsData
			self._obsData = self.getYadeData(yadeDataFiles)
			# loop over data assimilation steps
			for i in xrange(self._numSteps):
				self._likelihood[:,i], self._posterior[:,i], \
				self._ips[:,i], self._covs[:,i] = self.recursiveBayesian(i,self._proposal[:,i])			
		else:
			raise RuntimeError,"Calling Yade within python is not yet supported..."
		return self._ips, self._covs

	def getYadeData(self, yadeDataFiles):
		if 0 in self._yadeData.shape: raise RuntimeError,"number of Observations, samples or steps undefined!"
		obsData = np.zeros([self._numSteps, self._numObs])
		for i,f in enumerate(yadeDataFiles):
			yadeData = getKeysAndData(f)
			for j,key in enumerate(self._obsData.keys()):
				self._yadeData[:,i,j] = yadeData[key]
				obsData[:,j] = self._obsData[key]
		return obsData

	def recursiveBayesian(self,caliStep,proposal,iterNO=-1):
		likelihood = self.getLikelihood(caliStep)
		posterior = self.update(caliStep,likelihood,proposal)
		# get ensemble averages and coefficients of variance
		ips = np.zeros(self._numParams)
		covs = np.zeros(self._numParams)
		for i in xrange(self._numParams):
			# ensemble average
			ips[i] = self._smcSamples[iterNO][:,i].dot(posterior)
			# diagonal variance
			covs[i] = ((self._smcSamples[iterNO][:,i]-ips[i])**2).dot(posterior)
			# get coefficient of variance cov
			covs[i] = np.sqrt(covs[i])/ips[i]
		return likelihood, posterior, ips, covs
		
	def getLikelihood(self,caliStep):
		# state vector y_t = H(x_t)+Sigma_t
		stateVec = self._yadeData[caliStep,:,:].dot(self._obsMatrix)
		obsVec = self._obsData[caliStep,:]
		# row-wise substraction obsVec[numObs]-stateVec[numSamples,numObs]
		vecDiff = obsVec-stateVec
		Sigma = self.getCovMatrix(caliStep,self._obsWeights)
		invSigma = np.linalg.inv(Sigma)
		likelihood = np.zeros(self._numSamples)
		# compute likelihood = exp(-0.5*(y_t-H(x_t))*Sigma_t^{-1}*(y_t-H(x_t)))
		for i in xrange(self._numSamples):
			power = (vecDiff[i,:]).dot(invSigma.dot(vecDiff[i,:].T))
			likelihood[i] = np.exp(-0.5*power)
		# regularize likelihood
		likelihood /= np.sum(likelihood)
		return likelihood

	def update(self,caliStep,likelihood,proposal):
		# update posterior probability according to Bayes' rule
		posterior = np.zeros(self._numSamples)
		if caliStep == 0:
			posterior = likelihood*self._prior/proposal
		else: 
			posterior = self._posterior[:,caliStep-1]*likelihood*self._prior/proposal
			# regularize likelihood
		posterior /= np.sum(posterior)
		return posterior
		
	def getCovMatrix(self,caliStep,weights):
		Sigma = np.zeros([self._numObs,self._numObs])
		# scale observation data with normalized variance parameter to get covariance matrix
		for i in xrange(self._numObs):
			# give smaller weights for better agreement  
			Sigma[i,i] = self._sigma*weights[i]*self._obsData[caliStep,i]**2
		return Sigma

	def getInitParams(self, paramRanges, numSamples, thread):
		if not isinstance(paramRanges,dict): raise RuntimeError,"paramRanges should be a dictionary"
		self._numSamples = numSamples
		self._paramRanges = paramRanges
		self._numParams = len(paramRanges)
		names = self.getNames(); mins = []; maxs = []
		minsAndMix = paramRanges.values()
		for i,_ in enumerate(names):
			mins.append(min(minsAndMix[i]))
			maxs.append(max(minsAndMix[i]))
		print 'Parameters to be identified:', ", ".join(names), '\nMins:', mins, '\nMaxs:', maxs
		initSmcSamples, initSampleDataFile = initParamsTable(keys=names,maxs=maxs,mins=mins,num=numSamples,thread=thread)
		self._smcSamples.append(initSmcSamples)
		self._sampleDataFiles.append(initSampleDataFile)

	def getParamsFromTable(self, sampleDataFile, names, paramRanges, iterNO=-1):
		self._paramRanges = paramRanges
		self._numParams = len(paramRanges)
		if len(sampleDataFile) != 0: self._sampleDataFiles.append(sampleDataFile)
		else: raise RuntimeError,"Missing filename: file that contains smc samples cannot be found"
		if os.path.exists(sampleDataFile):
			smcSamples = np.genfromtxt(sampleDataFile,comments='!')[:,-self._numParams:]
			self._smcSamples.append(smcSamples)
			self._numSamples,_ = self._smcSamples[iterNO].shape
		else:
			yadeDataFiles = glob.glob(self._yadeDataDir+'/*txt')
			yadeDataFiles.sort()
			while len(yadeDataFiles) == 0:
				key = raw_input("No DEM filename has key, tell me the key...\n ")
				yadeDataFiles = glob.glob(self._yadeDataDir+'/*'+key+'*')
				yadeDataFiles.sort()
			smcSamples = np.zeros([len(yadeDataFiles),len(paramRanges)])
			for i,f in enumerate(yadeDataFiles):
				f = f.split('.txt')[0]
				_, key, E, mu_i, mu, k_r, mu_r = f.split('_')
				smcSamples[i,:] = eval(' '.join(['float('+name+'),' for name in names])[:-1])
			self._smcSamples.append(smcSamples)
			self._numSamples,_ = self._smcSamples[iterNO].shape
			
	def getObsDataFromFile(self,obsDataFile,obsCtrl):
		keysAndData = getKeysAndData(obsDataFile)
		# separate obsCtrl for controlling simulations from obsData
		obsCtrlData = keysAndData.pop(obsCtrl)
		numSteps = len(obsCtrlData)
		numObs = len(keysAndData.keys())
		return keysAndData, obsCtrlData, numObs, numSteps

	def resampleParams(self,caliStep,maxNumComponents=10,thread=4,iterNO=0):
		names = self.getNames()
		smcSamples = self._smcSamples[iterNO]
		numSamples = self._numSamples
		# posterior at caliStep is used as the proposal distribution
		proposal = self._posterior[:,caliStep]
		newSmcSamples, newSampleDataFile, gmm, maxNumComponents = \
			resampledParamsTable(keys=names,smcSamples=smcSamples,proposal=proposal,num=numSamples,thread=thread,maxNumComponents=maxNumComponents)
 		self._smcSamples.append(newSmcSamples)
		self._sampleDataFiles.append(newSampleDataFile)
		return gmm, maxNumComponents
	
	def getPosterior(self):
		return self._posterior

	def getSmcSamples(self):
		return self._smcSamples

	def getNumSteps(self):
		return self._numSteps

	def getEffectiveSampleSize(self):
		nEff = 1./sum(self.getPosterior()**2)
		return nEff/self._numSamples

	def getNames(self):
		return self._paramNames
	
	def getObsData(self):
		return np.hstack((self._obsCtrlData.reshape(self._numSamples,1),self._obsData))

	def trainGMMinTime(self,maxNumComponents,iterNO=0):
		gmmList = []
		smcSamples = self._smcSamples[iterNO]
		for i in xrange(self._numSteps):
			print 'Train DP mixture at time %i...'%i
			posterior = self._posterior[:,i]
			gmmList.append(getGMMFromPosterior(smcSamples,posterior,maxNumComponents))
		return gmmList
