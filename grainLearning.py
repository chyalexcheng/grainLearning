## Driver script to perform grainLearning
# set all grain learning dependent definitions in initialisation file (initialiseSimulation)!
import os
import sys 
import pickle
import tools
import numpy as np
from plotResults import *
import subprocess
from smc import *

class Logger(object):
    def __init__(self,logname):
        self.terminal = sys.stdout
        self.log = open(logname, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

# pickle name containing the grain learning initialisation information. Automatically changed when input_declaration is run 
pickle_name ="initialisationData.pkl"
# Read in which step of grainlearning is performed 
try:
  iterNO = int(sys.argv[1])
except:
  print("Exit grain learning since step number is missing in argument") 
  quit()
  
logname = 'pythonLog'+str(iterNO)+'.txt'
sys.stdout = Logger(logname)

# check step number: -1 = initialisation, 0 step 0, 1 step 1 etc. 
if(iterNO==-1):
    # Simulation number is 0 
    simNum = 0
    # if neccessary, delete file indicating that the simulation if finished 
    if os.path.exists('finishedGL.txt'):
       os.remove('finishedGL.txt')
    # generate parental folder for simulation results 
    with open(pickle_name,'rb') as f:
        baseFolder,yadeDataDir,yadeFile,simName,obsFileName,tableName,sigma,ess,obsWeights,maxNumOfIters,threads,paramNames,numParams,paramRanges,critRanges,obsCtrl,simDataKeys,alpha,numSamples,maxNumComponents,priorWeight,covType,s_t,igl_t,cse_t,icse_t, platform,software,software_version,glErrors,csErrors = pickle.load(f)
    cwd = os.getcwd()  
    path = cwd+"/"+yadeDataDir
    
    
    
    if not os.path.exists(path):
       print("No pre-existing folder", yadeDataDir)
       print('Generate folder',yadeDataDir)
       os.mkdir(path)     
    else: 
        num=1
        rename_path=path
        while  os.path.exists(rename_path):
           folder_name = yadeDataDir+"_old_"+str(num)
           rename_path = cwd+"/"+ folder_name
           num+=1
        print("Pre-existing folder", yadeDataDir, " renamed to ", folder_name)
        os.rename(path,rename_path)
        print('Generate folder',yadeDataDir)
        os.mkdir(path)    
       
    # Get mins and maxs of pre-defined ranges . Need to be provided seperately
    mins = []
    maxs = []
    for key in paramNames:
        mins.append(paramRanges[key][0])
        maxs.append(paramRanges[key][1]) 
    name = tableName+str(0)+'.txt'
    # instantiate the problem, define first table 
    tools.initParamsTable(paramNames, maxs, mins, num=numSamples, threads=threads, tableName=name)
    # move first table to folder 
    moveTable = "mv"+" "+tableName+str(0)+'.txt'+" "+yadeDataDir
    subprocess.call(moveTable, shell=True)  
    
    # start simulations (implemented in tools)
    name =  yadeDataDir+"/"+tableName+str(0)+'.txt'
    
    # Check if starting of new simulations suppressed
    startSimulations = True
    try:
        newSim = sys.argv[2]
        if(newSim=='No' or newSim=='no'):
            startSimulations=False
            print("No new simulations are started")
        else:
            print("Start new simulations")
    except:
        print("Start new simulations")    
    if startSimulations: tools.startSimulations(platform,software,name,yadeFile)
    
# normal grainlearning step   
else:  
 simNum=iterNO+1   
 # Check if initialisation file can be found 
 if not os.path.isfile(pickle_name):
     print("Initialisation pickle is not found. 'Run input_declaration'. Exit")
     sys.exit
 # Get data from initialisation file     
 with open(pickle_name,'rb') as f:
        baseFolder,yadeDataDir,yadeFile,simName,obsFileName,tableName,sigma,ess,obsWeights,maxNumOfIters,threads,paramNames,numParams,paramRanges,critRanges,obsCtrl,simDataKeys,alpha,numSamples,maxNumComponents,priorWeight,covType,s_t,igl_t,cse_t,icse_t, platform,software,software_version,glErrors,csErrors = pickle.load(f)
 # Move files from previous simulations into simulation folder (possible old folders are automatically deleted)
 cwd = os.getcwd() 
 newFolderFiles = cwd+"/"+yadeDataDir + '/iter' +str(iterNO) 
 if not os.path.exists(newFolderFiles):
   os.mkdir(newFolderFiles)
 moveFiles = "mv"+" "+simName+"*.npy"+" "+newFolderFiles
 subprocess.call(moveFiles, shell=True) 
 
 # Same for log files 
 newFolderLogs = cwd+"/"+yadeDataDir + '/log_iter' +str(iterNO) 
 if not os.path.exists(newFolderLogs): 
   os.mkdir(newFolderLogs)
 
 # Go to baseFolder where yade file is stored and the log files are generated   
 os.chdir(baseFolder)  
 moveLogs = "mv"+" "+"*.log"+" "+newFolderLogs
 subprocess.call(moveLogs, shell=True) 
 # Go back to default folder 
 os.chdir(cwd)
 
# Define seed for random number generator
 seed=simNum
 np.random.seed(seed)
# perform grain learning
 smcTest = smc(sigma, ess, obsWeights,
              yadeDataDir=yadeDataDir, threads=threads,
              obsCtrl=obsCtrl, simDataKeys=simDataKeys, simName=simName, obsFileName=obsFileName, seed=seed,
              loadSamples=True, runYadeInGL=False, standAlone=True)
 
 paramsFile = yadeDataDir+"/"+tableName+str(iterNO)+'.txt'
# load parameter samples
 smcTest.initParams(paramNames, paramRanges, numSamples, paramsFile=paramsFile, subDir='iter%i' % iterNO, simNum=simNum)

# initialize the weights
 smcTest.initialize(maxNumComponents, priorWeight, covType=covType)
# sequential Monte Carlo
 ips, covs = smcTest.run(iterNO=iterNO,reverse=iterNO % 2)
# get the parameter samples (ensemble) and posterior probability
 posterior = smcTest.getPosterior()
 smcSamples = smcTest.getSmcSamples()
 
 ## plot means of PDF over the parameters
 plotIPs(paramNames, ips.T, covs.T, smcTest.getNumSteps(), posterior, smcSamples[0])
 moveFile = "mv"+" "+"Means_over_parameters.pdf"+" "+newFolderFiles
 moveFile2 = "mv"+" "+"Coefficients_of_variance.pdf"+" "+newFolderFiles
 moveFile3 = "mv"+" "+"Posterior_PDF**.pdf"+" "+newFolderFiles
 subprocess.call(moveFile, shell=True)
 subprocess.call(moveFile2, shell=True)
 subprocess.call(moveFile3, shell=True)
 
 probableParams, IDs = smcTest.getMostProbableParams(5, -1)[0:2]
 print('Most probable parameters',probableParams)
 print('Associated sample IDs',IDs)
 print('Maximum sample size', ess)
 
 ##Compute different criteria for GL termination 
 # Get current grain learning and combined sample error (sample with highest probability) 
 gle, cse = smcTest.computeGrainLearningError(iterNO=iterNO,writeStochastic=True, writeNpy=True)
 # safe errors in pickle (override value if iteration has been restarted)
 if not len(glErrors)>iterNO:
     glErrors.append(gle)
 else:
     glErrors[iterNO]= gle
     
 if not len(csErrors)>iterNO:
     csErrors.append(cse)
 else:
     csErrors[iterNO]= cse    
     
 # Get grain learning and combined sample errors (sample with highest probability) of last step 
 gle_old = glErrors[iterNO-1]
 cse_old = csErrors[iterNO-1]
 
 
 # Compute error increments 
 delta_gle = gle_old-gle
 delta_cse = cse_old-cse 
 
 # In case of iteration zero iterNo = 0. Thus glErrors[iterNO-1]= glErrors[-1]. 
 #The value correspond to glErrors[0] and consequently delta_gle=0. To avoid termination after iteration catch this exception
 if iterNO==0:
     delta_gle =float('inf')
     delta_cse =float('inf')
 
 print("Grain learning mean absolute percentage error : ",gle)
 print("Increment in grain learning error: ",delta_gle)
 print("Combined sample error (for sample with highest probability) : ",cse)
 print("Increment in combined sample error (sample with highest probability) : ",delta_cse)
 
 # check if grain learning is donewith respect to Eq. 12 
 if smcTest.sigma<=s_t or delta_gle<=igl_t or cse<= cse_t or  delta_cse<=icse_t:
     with open('finishedGL.txt', 'w') as f: f.write("Grain Learning finished after iteration " +str(iterNO)+ "with sigma = "+str(smcTest.sigma))
     # Copy pickle file into folder 
     movePickle = "cp"+" "+pickle_name+" "+yadeDataDir
     subprocess.call(movePickle, shell=True) 
 else:    
   # resample parameters for next step 
   caliStep = -1
   name = yadeDataDir+"/"+tableName+str(simNum)+'.txt'
   print('yes')
   gmm, maxNumComponents = smcTest.resampleParams(caliStep=caliStep,paramRanges=critRanges,iterNO=iterNO,tableName=name,simNum=simNum)
 
  ## Plotting after resampling 
   # plot initial and resampled parameters
   plotAllSamples(smcTest.getSmcSamples(), smcTest.getNames())
   moveFile = "mv"+" "+"ResampledParameterSpace.pdf"+" "+newFolderFiles
   subprocess.call(moveFile, shell=True)

     # Check if starting of new simulations suppressed
   startSimulations = True
   try:
       newSim = sys.argv[2]
       if(newSim=='No' or newSim=='no'):
           startSimulations=False
           print("No new simulations are started")
       else:
           print("Start new simulations")
   except:
       print("Start new simulations")    
   if startSimulations: tools.startSimulations(platform,software,name,yadeFile)
 # Move logfile 
 moveFiles = "mv"+" "+ logname+" "+newFolderFiles
 subprocess.call(moveFiles, shell=True)  
 # change sigma 
 sigma = smcTest.sigma
 with open(pickle_name,'wb') as f:
    pickle.dump([baseFolder,yadeDataDir,yadeFile,simName,obsFileName,tableName,sigma,ess,obsWeights,maxNumOfIters,threads,paramNames,numParams,paramRanges,critRanges,obsCtrl,simDataKeys,alpha,numSamples,maxNumComponents,priorWeight,covType,s_t,igl_t,cse_t,icse_t, platform,software,software_version,glErrors,csErrors], f)
