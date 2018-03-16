""" Sequential Monte Carlo: the core of the calibration toolbox
	(https://en.wikipedia.org/wiki/Particle_filter)
"""

import sys
import os
import glob
import numpy as np
from tools import * 

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
		self._obsData, self._obsCtrlData, self._numObs, self._numSteps = self.getObsData(obsDataFile,obsCtrl)
		# assume all measurements are independent
		self._obsMatrix = np.identity(self._numObs)
		self._sampleDataFile = ''
		self._paramRanges = {}
		self._smcSamples = np.zeros(self._numSamples)
		# a flag that defines whether Yade is called within python
		self._standAlone = standAlone
		
	def initialize(self, paramRanges, numSamples, sampleDataFile='', thread=4, loadSamples=False):
		if loadSamples: self.getParamsFromTable(sampleDataFile, paramRanges)
		else: self.setParams(paramRanges, numSamples, thread)
		if self._standAlone:
			# simulation data
			self._yadeData = np.zeros([self._numSteps, self._numSamples, self._numObs])
			# identified optimal parameters
			self._ips = np.zeros([self._numParams, self._numSteps])
			self._covs = np.zeros([self._numParams, self._numSteps])
			self._posterior = np.zeros([self._numSamples, self._numSteps])
			self._likelihood = np.zeros([self._numSamples, self._numSteps])

	def run(self,skipDEM=False):
		if self._standAlone:
			if not skipDEM:
				# run DEM simulations in batch. 
				raw_input("*** Press Enter if the file name is correct... ***\n"+self._name\
						  +"\n(Add 'key%03d'%key of sampleDataFiles into DEM filenames)" )
				system(' '.join([self._yadeVersion, self._sampleDataFile, self._name]))
				print 'All simulations finished'
			else: print 'Skipping DEM simulations, read in data now'
			yadeDataFiles = glob.glob(self._yadeDataDir+'/*key*')
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
				self._ips[:,i], self._covs[:,i] = self.recursiveBayesian(i)
		else:
			raise RuntimeError,"Calling Yade within python is not yet supported..."

	def getYadeData(self, yadeDataFiles):
		if 0 in self._yadeData.shape: raise RuntimeError,"number of Observations, samples or steps undefined!"
		obsData = np.zeros([self._numSteps, self._numObs])
		for i,f in enumerate(yadeDataFiles):
			yadeData = getKeysAndData(f)
			for j,key in enumerate(self._obsData.keys()):
				self._yadeData[:,i,j] = yadeData[key]
				obsData[:,j] = self._obsData[key]
		return obsData

	def recursiveBayesian(self,caliStep):
		likelihood = self.getLikelihood(caliStep)
		posterior = self.update(caliStep,likelihood)
		# get ensemble averages and coefficients of variance
		ips = np.zeros(self._numParams)
		covs = np.zeros(self._numParams)
		for i in xrange(self._numParams):
			# ensemble average
			ips[i] = self._smcSamples[:,i].dot(posterior)
			# diagonal variance
			covs[i] = ((self._smcSamples[:,i]-ips[i])**2).dot(posterior)
			# get coefficient of variance cov
			covs[i] = np.sqrt(covs[i])/ips[i]
		return likelihood, posterior, ips, covs
		
	def getLikelihood(self,caliStep):
		# state vector y_t = H(x_t)+Sigma_t
		stateVec = self._yadeData[caliStep,:,:].dot(self._obsMatrix)
		obsVec = self._obsData[caliStep,:]
		# row-wise substraction obsVec[numObs]-stateVec[numSamples,numObs]
		vecDiff = obsVec-stateVec
		Sigma = self.getCovMatrix(self._obsWeights)
		invSigma = np.linalg.inv(Sigma)
		likelihood = np.zeros(self._numSamples)
		# compute likelihood = exp(-0.5*(y_t-H(x_t))*Sigma_t^{-1}*(y_t-H(x_t)))
		for i in xrange(self._numSamples):
			power = (vecDiff[i,:]).dot(invSigma.dot(vecDiff[i,:].T))
			likelihood[i] = np.exp(-0.5*power)
		# regularize likelihood
		likelihood /= np.sum(likelihood)
		return likelihood

	def update(self,caliStep,likelihood):
		# update posterior probability according to Bayes' rule
		posterior = np.zeros(self._numSamples)
		if caliStep == 0:
			posterior = likelihood
		else: 
			posterior = self._posterior[:,caliStep-1]*likelihood
			# regularize likelihood
			posterior /= np.sum(posterior)
		return posterior
		
	def getCovMatrix(self,weights):
		Sigma = np.zeros([self._numObs,self._numObs])
		# scale observation data with normalized variance parameter to get covariance matrix
		for i in xrange(self._numObs):
			# give smaller weights for better agreement  
			Sigma[i,i] = self._sigma*weights[i]*np.mean(self._obsData[:,i])**2
		return Sigma

	def setParams(self, paramRanges, numSamples, thread):
		if not isinstance(paramRanges,dict): raise RuntimeError,"paramRanges should be a dictionary"
		self._numSamples = numSamples
		self._paramRanges = paramRanges
		self._numParams = len(paramRanges)
		names = paramRanges.keys(); mins = []; maxs = []
		minsAndMix = paramRanges.values()
		for i,_ in enumerate(names):
			mins.append(min(minsAndMix[i]))
			maxs.append(max(minsAndMix[i]))
		print 'Parameters to be identified:', ", ".join(names), '\nMins:', mins, '\nMaxs:', maxs
		self._smcSamples, self._sampleDataFile = paramsTable(keys=names,maxs=maxs,mins=mins,num=numSamples,thread=thread)

	def getParamsFromTable(self, sampleDataFile, paramRanges):
		if len(sampleDataFile) != 0: self._sampleDataFile = sampleDataFile
		else: raise RuntimeError,"Missing filename: file that contains smc samples cannot be found"
		self._paramRanges = paramRanges
		self._numParams = len(paramRanges)
		self._smcSamples = np.genfromtxt(sampleDataFile,comments='!')[:,-self._numParams:]
		self._numSamples,_ = self._smcSamples.shape

	def getObsData(self,obsDataFile,obsCtrl):
		keysAndData = getKeysAndData(obsDataFile)
		# separate obsCtrl for controlling simulations from obsData
		obsCtrlData = keysAndData.pop(obsCtrl)
		numSteps = len(obsCtrlData)
		numObs = len(keysAndData.keys())
		return keysAndData, obsCtrlData, numObs, numSteps
