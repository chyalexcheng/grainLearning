""" Author: Hongyang Cheng <chyalexcheng@gmail.com>
	 An interface to generate parameter table with Halton sequence
	 (requires ghalton library: https://pypi.python.org/pypi/ghalton)
"""

from math import *
import ghalton
import numpy as np
from resample import *
from sklearn import mixture

def initParamsTable(keys,maxs,mins,num=100,threads=4,tableName='smcTable.txt'):
	"""
   :param dim: type integer, number of parameters
   :param num: type integer, number of sampling points for Monte Carlo Simulation
   :param threads: type integer, number of thread for each parallel simulation
   :param maxs: type tuples, maximums ranges of parameters
   :param mins: type tuples, minimums ranges of parameters
   :param keys: type strings, names of parameters
	"""
	dim = len(keys)
	sequencer = ghalton.Halton(dim)
	table = sequencer.get(num)
	for i in xrange(dim):
		for j in xrange(num):
			mean = .5*(maxs[i]+mins[i])
			std  = .5*(maxs[i]-mins[i])
			table[j][i] = mean+(table[j][i]-.5)*2*std
	# write parameters in the format for Yade batch mode
	writeToTable(tableName,table,dim,num,threads,keys)
	return table, tableName
	
def writeToTable(tableName,table,dim,num,thread,keys):
	# output parameter table with thread number for each Yade simulation session
	fout = file(tableName,'w')
	fout.write(' '.join(['!OMP_NUM_THREADS','key']+keys+['\n']))
	for j in xrange(num):
		fout.write(' '.join(['%2i'%thread,'%9i'%j]+['%15.5e'%table[j][i] for i in xrange(dim)]+['\n']))
	fout.close()
	
def getKeysAndData(fileName):
	data = np.genfromtxt(fileName)
	fopen = open(fileName,'r')
	keys = (fopen.read().splitlines()[0]).split('\t\t')
	if '#' in keys: keys.remove('#')
	keysAndData = {}
	for key in keys: keysAndData[key] = data[:,keys.index(key)]
	return keysAndData

def resampledParamsTable(keys,smcSamples,proposal,num=100,thread=4,maxNumComponents=10,priorWeight=1e3,tableName='smcTableNew.txt'):
	dim = len(keys)
	# resample parameters from a proposal PDF
	ResampleIndices = unWeighted_resample(proposal,10*num)
	smcNewSamples = smcSamples[ResampleIndices]
	# normalize parameter samples
	sampleMaxs = np.zeros(smcSamples.shape[1])
	for i in range(sampleMaxs.shape[0]):
		sampleMaxs[i] = max(smcNewSamples[:,i])
		smcNewSamples[:,i] /= sampleMaxs[i]
	# regenerate new SMC samples from Bayesian gaussian mixture model
	# details on http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
	gmm = mixture.BayesianGaussianMixture(n_components=maxNumComponents,weight_concentration_prior=priorWeight,covariance_type='full',tol = 1e-5,max_iter=int(1e5),n_init=100)
	gmm.fit(smcNewSamples)
	smcNewSamples, _ = gmm.sample(num)
	# scale resampled parameters back to their right units
	for i in range(sampleMaxs.shape[0]): smcNewSamples[:,i] *= sampleMaxs[i]
	# write parameters in the format for Yade batch mode
	writeToTable(tableName,smcNewSamples,dim,num,thread,keys)
	return smcNewSamples, tableName, gmm, maxNumComponents

def getGMMFromPosterior(smcSamples,posterior,priorWeight):
	# resample parameters from a proposal PDF
	ResampleIndices = residual_resample(posterior)
	smcNewSamples = smcSamples[ResampleIndices]
	n_components = int(self._numSamples/5)
	gmm = mixture.BayesianGaussianMixture(n_components=n_components,weight_concentration_prior=priorWeight,covariance_type='full',tol = 1e-5,max_iter=int(1e5),n_init=100)
	gmm.fit(smcNewSamples)
	return gmm	

def get_pool(mpi=False,threads=1):
   if mpi: # using MPI
      from mpipool import MPIPool
      pool = MPIPool()
      pool.start()
      if not pool.is_master():
         sys.exit(0)
   elif threads>1: # using multiprocessing
      from multiprocessing import Pool
      pool = Pool(processes=threads,maxtasksperchild=10)
   else:
      raise RuntimeError,"Wrong arguments: either mpi=True or threads>1."
   return pool
