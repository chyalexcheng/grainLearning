#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:07:38 2021

@author: Philipp Hartmann
"""
import sys
import pickle
import subprocess
from colorama import Fore

def applyInputChanges(pickle_name):
    # load pickle with all information
 with open(pickle_name,'rb') as f:
        baseFolder,yadeDataDir,yadeFile,simName,obsFileName,tableName,sigma,ess,obsWeights,maxNumOfIters,threads,paramNames,numParams,paramRanges,critRanges,obsCtrl,simDataKeys,alpha,numSamples,maxNumComponents,priorWeight,covType,s_t,igl_t,cse_t,icse_t, platform,software,software_version,glErrors,csErrors = pickle.load(f)
        
 ## Adjust scripts 
 # 1. Grain learning driver script 
 # 2. File for model evaluations (e.g. yade-file)
 # 3. Platform dependent shell scripts (starting simulations)
 
 ## GL driver script 
 # adjust pickle name 
 command = 'sed -i \'s@pickle_name =\'.*\'@pickle_name =\"'+pickle_name+'\"@\' grainLearning.py'
 subprocess.call(command, shell=True) 
 
 ## Yade script/ Simulation script 
 # adjust name (path) of observation File. Important: Name has to be "obsFile" in simulation file
 command = 'sed -i \'s@obsFile =\'.*\'@obsFile =\"'+obsFileName+'\"@\' '+yadeFile
 subprocess.call(command, shell=True) 
 # change baseFolder 
 command = 'sed -i \'s@baseFolder =\'.*\'@baseFolder =\"'+baseFolder+'\"@\' '+yadeFile
 subprocess.call(command, shell=True) 
 # shell scripts   
 if platform=='desktop':
     if software=='yade':
         # Adjust number of cpus for model evaluation
         command = 'sed -i \'s/cpus=\'.*\'/cpus=\"'+str(threads)+'\"/\' platform_shells/desktop/yadeDesktop.sh'
         subprocess.call(command, shell=True)  
         # Adjust yade version 
         command = 'sed -i \'s@yadeVersion=\'.*\'@yadeVersion=\"'+software_version+'\"@\' platform_shells/desktop/yadeDesktop.sh'
         subprocess.call(command, shell=True)  
     else:
         print(Fore.RED +"Chosen 'software' has not been implemented yet. Check 'applyInputChanges()' in 'initialisationTools.py'")
         sys.exit
         
 elif platform=='aws':  
     if software=='yade':
         # Adjust number of cpus for model evaluation
         command = 'sed -i \'s/cpus=\'.*\'/cpus=\"'+str(threads)+'\"/\' platform_shells/aws/yadeAWS.sh'
         subprocess.call(command, shell=True)  
         # Adjust yade version
         command = 'sed -i \'s@yadeVersion=\'.*\'@yadeVersion=\"'+software_version+'\"@\' platform_shells/aws/yadeAWS.sh'
         subprocess.call(command, shell=True)               
     else:
         print(Fore.RED +"Chosen 'software' has not been implemented yet. Check 'applyInputChanges()' in 'initialisationTools.py'")
         sys.exit
  
 elif platform=='rcg':  
     if software=='yade':
         # Adjust number of cpus for model evaluation
         command = 'sed -i \'s/cpus=\'.*\'/cpus=\"'+str(threads)+'\"/\' platform_shells/rcg/yadeRCG.sh'
         subprocess.call(command, shell=True) 
         command = 'sed -i \'s/cpus=\'.*\'/cpus=\"'+str(threads)+'\"/\' platform_shells/rcg/runSingleYade.sh'
         subprocess.call(command, shell=True) 
         # Adjust yade version
         command = 'sed -i \'s@yadeVersion=\'.*\'@yadeVersion=\"'+software_version+'\"@\' platform_shells/rcg/yadeRCG.sh'
         subprocess.call(command, shell=True)   
     else:
         print(Fore.RED +"Chosen 'software' has not been implemented yet. Check 'applyInputChanges()' in 'initialisationTools.py'")
         sys.exit         

## Function to load the default values of grain learning parameters (calibration specific)
def initialise_predefined_calibration(calibration):
    if(calibration=="Triax_Syn"):    
      baseFolder='TriaxSynthetic'  
      yadeFile='TriaxSynthetic.py'
      obsFileName='syntheticObsData.txt' # File of observation data for calibration  
      simName='triax' # Filename for yade simulation results
      paramRanges = {} 
      paramRanges['E_m'] = [7,11]       
      paramRanges['v'] = [0,0.5]
      paramRanges['kr'] = [0,1]
      paramRanges['eta'] = [0,1]
      paramRanges['mu'] = [0,60]  
      # key for simulation control
      obsCtrl = "e_z"
      simDataKeys =["e_v","s33_over_s11"]
      obsWeights = [0.5,0.5]  
    
    if(calibration=="Triax_Exp"):  
      baseFolder='TriaxExperimental'   
      yadeFile='TriaxFinal_CL.py'
      obsFileName='Test_0.5.data' # File of observation data for calibration 
      simName='triax'     # Filename for yade simulation results  
      paramRanges = {} 
      paramRanges['E_m'] = [7,11]       
      paramRanges['v'] = [0,0.5]
      paramRanges['kr'] = [0,1]
      paramRanges['eta'] = [0,1]
      paramRanges['mu'] = [0,60]  
    
      obsCtrl = "e_z"
      simDataKeys =["e_v","s33_over_s11"]
      obsWeights = [1.0,1.0]  
    
    if(calibration=="Collision"): 
      baseFolder = 'Collision'    
      yadeFile='Collision.py'
      obsFileName='collisionOrg.dat'  
      simName='2particle'
      paramRanges = {} 
      paramRanges['E_m'] = [7,11]       
      paramRanges['nu'] = [0,0.5]  
      # key for simulation control
      obsCtrl = 'u'
      # key for output data
      simDataKeys = ['f']    
      obsWeights = [1.0] 
    
    # Perform safety checks 
    if not len(simDataKeys)==len(obsWeights):
        print(Fore.RED +'Number of obsWeights not in line with number of observations. Check file "initialisationTools"')  
        sys.exit
    if not isinstance(obsCtrl, str):
        print(Fore.RED +'Only a single observation control is valid as string. Check file "initialisationTools"')  
        sys.exit      
    # return values 
    return baseFolder, yadeFile, obsFileName, simName, paramRanges, obsCtrl, simDataKeys, obsWeights