""" Author: Hongyang Cheng <chyalexcheng@gmail.com>
     An interface to generate parameter table with Halton sequence
     (requires ghalton library: https://pypi.python.org/pypi/ghalton)
"""

from math import *
import ghalton
import sys
import os
import numpy as np
from resample import *
from colorama import Fore
from sklearn import mixture
import subprocess

def startSimulations(platform,software,tableName,fileName):   
 #platform desktop, aws or rcg    # software so far only yade 
 argument= tableName+" "+fileName
 if platform=='desktop':
     # Definition where shell script can be found
     path_to_shell = os.getcwd()+'/platform_shells/desktop' 
     if software=='yade':
         command = 'sh '+path_to_shell+'/yadeDesktop.sh'+" "+argument  
         subprocess.call(command, shell=True)  
     else:
         print(Fore.RED +"Chosen 'software' has not been implemented yet. Check 'startSimulations()' in 'tools.py'")
         sys.exit
         
 elif platform=='aws':  
     path_to_shell = os.getcwd()+'/platform_shells/aws' 
     if software=='yade':
         command = 'sh '+path_to_shell+'/yadeAWS.sh'+" "+argument  
         subprocess.call(command, shell=True)  
     else:
         print(Fore.RED +"Chosen 'software' has not been implemented yet. Check 'startSimulations()' in 'tools.py'")
         sys.exit
  
 elif platform=='rcg':  
     path_to_shell = os.getcwd()+'/platform_shells/rcg' 
     if software=='yade':
         command = 'sh '+path_to_shell+'/yadeRCG.sh'+" "+argument  
         subprocess.call(command, shell=True)  
     else:
         print(Fore.RED +"Chosen 'software' has not been implemented yet. Check 'startSimulations()' in 'tools.py'")
         sys.exit         
 else:
  print('Exit code. Hardware for yade simulations not properly defined')
  quit()




def initParamsTable(keys, maxs, mins, num=100, threads=4, tableName='smcTable0.txt', simNum=0):
    """
    Generate initial parameter samples using a halton sequence
    and write the samples into a text file

    :param keys: list of strings, names of parameters

    :param maxs: list of floats, upper bounds of parameter values

    :param mins: list of floats, lower bounds of parameter values

    :param num: int, default=100, number of samples for Sequential Monte Carlo

    :param threads: int, default=4, number of threads for each model evaluation

    :param tableName: string, Name of the parameter table

    :return:
        table: ndarray of shape (num, len(keys)), initial parameter samples

        tableName: string, default='smcTable.txt'
    """
    print(tableName)
    dim = len(keys)
    sequencer = ghalton.Halton(dim)
    table = sequencer.get(num)
    for i in range(dim):
        for j in range(num):
            mean = .5 * (maxs[i] + mins[i])
            std = .5 * (maxs[i] - mins[i])
            table[j][i] = mean + (table[j][i] - .5) * 2 * std
    # write parameters in the format for Yade batch mode
    writeToTable(tableName, table, dim, num, threads, keys, simNum)
    return np.array(table), tableName


def writeToTable(tableName, table, dim, num, threads, keys,simNum):
    """
    write parameter samples into a text file in order to run Yade in batch mode
    """
    
    # Computation of decimal number for unique key 
    magn = floor(log(num, 10))+1
    #iterNum = 0
    
    fout = open(tableName, 'w')
    fout.write(' '.join(['!OMP_NUM_THREADS', 'description', 'key'] + keys + ['\n']))
    for j in range(num):
       description = 'Iter'+str(simNum)+'-Sample'+str(j).zfill(magn)
       fout.write(' '.join(['%2i' % threads] + [description] + ['%9i' % j] + ['%20.10e' % table[j][i] for i in range(dim)] + ['\n']))
    fout.close()


def getKeysAndData(fileName):
    """
    Get keys and corresponding data sequence from a Yade output file

    :param fileName: string

    :return: keysAndData: dictionary
    """
    data = np.genfromtxt(fileName)
    fopen = open(fileName, 'r')
    keys = (fopen.read().splitlines()[0]).split('\t\t')
    if '#' in keys: keys.remove('#')
    keysAndData = {}
    for key in keys:
        if '#' in key: keyNoHash = key.split(' ')[-1]
        else: keyNoHash = key
        keysAndData[keyNoHash] = data[:, keys.index(key)]
    return keysAndData


def resampledParamsTable(keys, smcSamples, proposal, ranges, num=100, threads=4, maxNumComponents=10, priorWeight=0,
                         covType='full', tableName='smcTableNew.txt',seed=0,simNum=0):
    """
    Resample parameters using a variational Gaussian mixture model
    and write the samples into a text file
    
    :param keys: list of strings
        names of parameters

    :param smcSamples: ndarray of shape (num, len(keys))
        current parameter samples

    :param proposal: ndarray of shape num
        proposal probability distribution associated to current parameter samples

    :param num: int
        number of samples for Sequential Monte Carlo

    :param threads: int

    :param maxNumComponents: int, default=num/10

    :param priorWeight: float, default=1./maxNumComponents
        weight_concentration_prior of the BayesianGaussianMixture class
        The dirichlet concentration of each component on the weight distribution (Dirichlet).
        This is commonly called gamma in the literature.
        The higher concentration puts more mass in the center and will lead to more components being active,
        while a lower concentration parameter will lead to more mass at the edge of the mixture weights simplex.
        (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)

    :param covType: string, default='full'
        covariance_type of the BayesianGaussianMixture class
        String describing the type of covariance parameters to use. Must be one of:
        'full' (each component has its own general covariance matrix),
        'tied' (all components share the same general covariance matrix),
        'diag' (each component has its own diagonal covariance matrix),
        'spherical' (each component has its own single variance).
        (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)

    :param tableName: string, default='smcTableNew.txt'
        Name of the parameter table

    :return:
        newSMcSamples: ndarray of shape (num, len(keys))
            parameter samples for the next iteration

        tableName: string, default='smcTableNew.txt'
            Name of the parameter table

        gmm: BayesianGaussianMixture
            A variational Gaussian mixture model trained with current parameter samples and proposal probabilities

        maxNumComponents: int
            Number of sufficient Gaussian components for the mixture model
            (should be smaller than the input maxNumComponents)
    """

    dim = len(keys)
    # resample parameters from a proposal probability distribution
    ResampleIndices = unWeighted_resample(proposal, 10 * num)
    newSMcSamples = smcSamples[ResampleIndices]

    # normalize parameter samples
    sampleMaxs = np.zeros(smcSamples.shape[1])
    for i in range(sampleMaxs.shape[0]):
        sampleMaxs[i] = max(newSMcSamples[:, i])
        newSMcSamples[:, i] /= sampleMaxs[i]

    # regenerate new SMC samples from Bayesian gaussian mixture model
    # details on http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
    gmm = mixture.BayesianGaussianMixture(n_components=maxNumComponents, weight_concentration_prior=priorWeight,
                                          covariance_type=covType, tol=1e-5, max_iter=int(1e5), n_init=100,random_state=seed)
    gmm.fit(newSMcSamples)
    newSMcSamples, _ = gmm.sample(num)

    # ~ while num <= smcSamples.shape[0]:
        # ~ newSMcSamples, _ = gmm.sample(num)
        # ~ delParamIDs = []
        # ~ for i, param in enumerate(newSMcSamples):
            # ~ for j, name in enumerate(keys):
                # ~ if not (ranges[name][0] < param[j]*sampleMaxs[j] < ranges[name][1]):
                    # ~ delParamIDs.append(i)
                    # ~ break
        # ~ if not delParamIDs:
            # ~ print("Is empty")
            # ~ break
        # ~ newSMcSamples = np.delete(newSMcSamples, delParamIDs, 0)
        # ~ currentSampleSize = newSMcSamples.shape[0]
        # ~ num *= int(sampleSize/currentSampleSize)
    # check if parameters in predifined ranges. If not replace it randomly
    for i in range(num):
         for jj in range(len(keys)):
             name = keys[jj]
             val =  newSMcSamples[i, jj] #param[j]
             while not (ranges[name][0] <= val*sampleMaxs[jj] <= ranges[name][1]):
                 print('Parameter ',keys[jj] ,'in sample ',i, 'outside critical range')
                 k= np.random.randint(0,num)
                 newSMcSamples[i, jj] = newSMcSamples[k, jj]
                 val = newSMcSamples[i, jj]
           
    # scale resampled parameters back to their right units
    for i in range(sampleMaxs.shape[0]): newSMcSamples[:, i] *= sampleMaxs[i]  
    # write parameters in the format for Yade batch mode
    writeToTable(tableName, newSMcSamples, dim, num, threads, keys,simNum)
    return newSMcSamples, tableName, gmm, maxNumComponents


def getGMMFromPosterior(smcSamples, posterior, n_components, priorWeight, covType='full',seed=0):
    """
    Train a Gaussian mixture model from the posterior distribution
    """
    ResampleIndices = residual_resample(posterior)
    newSMcSamples = smcSamples[ResampleIndices]
    gmm = mixture.BayesianGaussianMixture(n_components=n_components, weight_concentration_prior=priorWeight,
                                          covariance_type=covType, tol=1e-5, max_iter=int(1e5), n_init=100,random_state=seed)
    gmm.fit(newSMcSamples)
    return gmm


def get_pool(mpi=False, threads=1):
    """
    Create a thread pool for paralleling DEM simulations within GrainLearning

    :param mpi: bool, default=False

    :param threads: int, default=1
    """
    if mpi:  # using MPI
        from mpipool import MPIPool
        pool = MPIPool()
        pool.start()
        if not pool.is_master():
            sys.exit(0)
    elif threads > 1:  # using multiprocessing
        from multiprocessing import Pool
        pool = Pool(processes=threads, maxtasksperchild=10)
    else:
        raise RuntimeError("Wrong arguments: either mpi=True or threads>1.")
    return pool
