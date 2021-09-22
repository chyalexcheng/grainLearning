"""
Author: Philipp Hartmann
"""

import numpy as np
import sys
import pickle
import initialisationTools
from colorama import Fore

## Please provide your Grain Learning input data. 
# Defaults are set for the calibration of micro-mechanical parameters for Cundell linear model 
# on the basis of triaxial compression test using experimental data. Model evaluations are performed
# with Yade and desktop-work station with pre-compiled yade version  

## Platform  information
# Define which platform should be used 
platform = 'desktop'   # Implemented are 'desktop', 'rcg', 'aws'
# Set number of threads per simulation (apply for all possible platforms)
ncpus=1 

## Model evaluation / software information
# Define software to be used. Is needed to start the model evaluation with the correct script
software = 'yade'
# Define software version, i.e. command required to run software (e.g. 'yade' for the pre-compiled yade version)
software_version = 'yade'


# Folder name in which the results will be moved
yadeDataDir= 'cali_results'
# Table 
tableName="samples"
 
# Filename for pickle (in this pickle all the GL initialisation data is saved)
pickle_name = 'initialisationData.pkl'

## Define calibration/optimisation problem (easily extendable later on)
calibration = 'Collision'#'user_defined' #'Triax_Syn'    # Available 'Triax_Exp', 'Triax_Syn', 'Collision' and 'user_defined'


def user_defined_initialisation():  ## Default values are for collision example
    # folder in which observation data as well as yade file is provided. Empty string if data is put in main folder 
    baseFolder = 'Collision'   
    # file for model evaluation  
    yadeFile='Collision.py'
    # file which contains the 'target' observation data 
    obsFileName='collisionOrg.dat'
    # Filename for simulation results (has to be the same as defined in yadeFile for the generation of .npy files)
    simName='2particle'
    # Define parameters and pre-defined ranges 
    paramRanges = {}  
    paramRanges['E_m'] = [7,11]
    paramRanges['nu'] = [0,0.5]
    # key for simulation control data (has to be part of obsFileName)
    obsCtrl = "u"
    # define observations and associated weights (has to be part of obsFileName)
    simDataKeys =["f"]
    obsWeights = [1]    
    # Perform safety checks 
    if not len(simDataKeys)==len(obsWeights):
        print(Fore.RED +'Number of obsWeights not in line with number of observations. Check file "initialisationTools"')  
        sys.exit
    if not isinstance(obsCtrl, str):
        print(Fore.RED +'Only a single observation control is valid as string. Check file "initialisationTools"')  
        sys.exit 
    # return all defined values 
    return baseFolder, yadeFile, obsFileName, simName, paramRanges, obsCtrl, simDataKeys, obsWeights


if calibration == 'Triax_Exp' or calibration == 'Triax_Syn' or calibration == 'Collision' or calibration == 'user_defined':
    print("Initialise parameters and corresponding ranges for ", calibration) 
    if calibration == 'user_defined':
        baseFolder, yadeFile, obsFileName, simName, paramRanges, obsCtrl, simDataKeys, obsWeights = user_defined_initialisation()
    else:
        baseFolder, yadeFile, obsFileName, simName, paramRanges, obsCtrl, simDataKeys, obsWeights = initialisationTools.initialise_predefined_calibration(calibration)
else:
    print(Fore.RED +"Please check the definition of the variable 'calibration'")
    sys.exit

# Concatenation of filenames with respect to base folder
yadeDataDir=baseFolder+"/"+yadeDataDir
yadeFile=baseFolder+"/"+yadeFile
obsFileName=baseFolder+"/"+obsFileName

# Get parameter names and number of parameters  
paramNames=[*paramRanges]
numParams = len(paramNames)

## Grain learning parameters 
# alpha value to determine the sample size 
alpha = 10
# compute number of samples per iteration (e.g., num1D * N * logN for quasi-Sequential Monte Carlo)
numSamples = int(alpha * numParams * np.log(numParams))

# Thresholds used in termination criteria
s_t = 0.01 # normalised covariance coefficient  
igl_t = 0.2  # increment grain learning error %
cse_t = 2  # combined sample error %
icse_t = 0.2  # increment combined sample error %

## No further user interaction is required. Don't change the code below
# upper limit of the normalized covariance coefficient
sigma = 1.0 
# target effective sample size
ess = 0.3
 # set the maximum Gaussian components and prior weight
maxNumComponents = int(numSamples / 10)
priorWeight = 1. / maxNumComponents
covType = 'tied'     
# Possibility to implement critical ranges, i.e. ranges in which parameters are allowed to be after resampling. Default are the predefined ranges
critRanges =  paramRanges

# save pickle 
threads=ncpus
maxNumOfIters=10 # unused parameter 
glErrors=[] # list of grain learning errors (necessary for termination criteria)
csErrors=[] # list of combined sample errors (necessary for termination criteria)
with open(pickle_name,'wb') as f:
    pickle.dump([baseFolder, yadeDataDir,yadeFile,simName,obsFileName,tableName,sigma,ess,obsWeights,maxNumOfIters,threads,paramNames,numParams,paramRanges,critRanges,obsCtrl,simDataKeys,alpha,numSamples,maxNumComponents,priorWeight,covType,s_t,igl_t,cse_t,icse_t, platform,software,software_version,glErrors,csErrors], f)
    
# push initialisation, i.e. apply definitions  
initialisationTools.applyInputChanges(pickle_name)


