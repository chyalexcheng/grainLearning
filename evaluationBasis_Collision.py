#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:19:41 2021

@author: Philipp Hartmann
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

folderName ='Collision/cali_results'
data = {} 

current = os.getcwd()
folders =os.listdir(current+'/'+folderName+'/collectiveNPY')
matching = [s for s in folders if  'collectiveInformation' in s]  
matching.sort(key=natural_keys)

# loop over all iterations and get all data 
num = 0
for i in matching:
    file = folderName+'/collectiveNPY/'+i
    a = np.load(file,allow_pickle=True).item()
    data[str(num)]=a
    num+=1

## Save and load the information above 
# np.save('data_Collision',data)
# data = np.load('data_Collision.npy',allow_pickle=True).item()   

## Data storage  
# The data for the i-iteration is stored in data[i] 
# Check data[0].keys() to see all available data 
    
## Use the data for plotting: Examples below
fs = 8 
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'lines.linewidth': 2})    

# 1. GL error over iterations   
glErrors = []    
for i in data.keys():
    glErrors.append(data[i]['gle'])
    
iters = range(0, len(glErrors))
plt.figure('Gl error',figsize=(fs,fs))
plt.plot(iters,glErrors,'*-')  
plt.ylabel('GL error [%]')
plt.xlabel('Iteration number')
    
# 2. Result of most probable sample in final iteration    
final = data[str(len(data.keys())-1)]
# get ID of sample related to the most probable parameters 
ids = final['ID'][0]
# material parameters
E_m = final['E_m'][0]
nu = final['nu'][0]

# get keys (names) of observation and observation control data  
obsNames = final['obsKeys']
obsCtrlName = final['obsCtrlKey']

# forces of associated sample
predictionsBest = final['YadeData'][:,ids,:]  # shape of final['YadeData']: (vector of model prediction, sampleID, observation number)
obsNumber=0
f = predictionsBest[:,obsNumber] # 0 because we only have 1 observation vector (forces)
# controlData
u= final['obsCtrlData']
# obsData
obsData= final['obsData']
# combined sample errors 
cse = final['CombinedSampleErrors'][0]

print('The calibrated material parameters of the most probable sample',ids,'are E_m:',E_m,'and nu=', nu,'. The combined sample error is',cse,'%')

# Force - displacement 
plt.figure('Force-displacement',figsize=(2*fs,fs))
plt.scatter(u,obsData, label='Syn. data',color='k')
plt.plot(u,f, label='Calibration')
plt.ylabel(obsNames[obsNumber])
plt.xlabel(obsCtrlName) 
plt.legend()


#plt.savefig("test.pdf",transparent = True,bbox_inches='tight')