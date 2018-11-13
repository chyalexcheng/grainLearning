""" Author: Hongyang Cheng <chyalexcheng@gmail>
 
	 A calibration tool for discrete element models of granular materials
	 using the particle filter
	 
	 Monte Carlo simulations are conducted with YADE. 
	 An initial 'stress-free' granular packing, experimental data 
	 (e.g., triaxial response) and parameter sets of DEM models 
	 are needed. Each parameter set is referred as a 'particle' 
	 which is generated from Halton sequence. The numbers of DEM 
	 simulation and experimental measurement steps must be the same. 
	 The identified parameters come in the forms of probability density
	 functions if the covariance is chosen properly.
"""    
	
from tableGenerator import paramsTable
from plotResults import *
import numpy as np
from os import system
from os import listdir
from os import path
import matplotlib.pyplot as plt

#########################################################
##  inputs for parameter identification of DEM models  ##
#########################################################

""" param names: type strings, names of parameters
    param maxs: type tuples, maximums ranges of parameters
    param mins: type tuples, minimums ranges of parameters
    param nSample: type integer, number of sampling points for Monte Carlo simulation
    param thread: type integer, number of thread for each parallel simulation
    param nStep: type integer, number of simulation and measurement steps
    param nObs: type integer, number of independent measurement points
    param cov: type float, used in covariance matrix of the error terms
    param mcsData: type boolean, whether run parallel DEM simulations or not
"""

# parameter names
names = ['E', 'v', 'kr', 'eta', 'mu']
print " ".join(['Parameters to be identified:']+names)

# whether a Monte Carlo simulation database is available
mcsData = raw_input("Use an existing Monte Carlo simulation database? (Y/N)\n")
if mcsData == 'Y': 
	mcsData = True
else: mcsData = False

# get number of measurement points and steps
obs = np.genfromtxt('obsdata.dat')
nStep = len(obs)
nObs  = len(obs[0])
	
####################################
##  run parallel DEM simulations  ##
####################################

## if no available database
if not mcsData:
	# upper bounds of parameter ranges
	string_input = raw_input("Upper bounds of parameter ranges: \n e.g., 5.5e8 0.2 0.6 0.2 31 \n")
	maxs = string_input.split()
	maxs = [float(maxValue) for maxValue in maxs]
	
	# lower bounds of parameter ranges
	string_input = raw_input("Lower bounds of parameter ranges: \n e.g., 5.4e8 0.0 0.5 0.1 29 \n")
	mins = string_input.split()
	mins = [float(minValue) for minValue in mins]

	# set number of samples for Monte Carlo simulation and threads for each YADE session
	nSample = input("Number of sampling points: \n e.g., 100 \n")
	thread = input("threads for each YADE session: \n e.g., 4 \n")

	# generate parameter table with halton sequence
	paramsTable(names,maxs,mins,nSample,thread)
	# run parallel YADE simulations
	system("yadedaily-batch table.dat ../mcTriax.py")
else:
	print "Use an existing Monte Carlo simulation database"
	nSample = len(listdir('mcSimulations'))
	
#########################################
## input files for particle filtering  ##
#########################################

# Monte Carlo simulation data file
fout = file('MCS.dat','w')
for i in xrange(nSample):
   fout.write(str(i+1)+'\n')
   data = np.load('mcSimulations/'+str(i)+'.npy').item()
   for j in xrange(nStep):
      fout.write('%15.5e'%data['e_r'][j]+'%15.5e'%data['s33_over_s11'][j]+'%15.5e'%data['e_v'][j]+'\n')
fout.close()

# control variables for PF.exe
fout = file('control_parameter.txt','w')
fout.write('%9i'%nStep     +' ! simulation step\n')
fout.write('%9i'%nSample   +' ! nsample in particle_filter.f90\n')
fout.write('%9i'%nObs      +' ! number of measurement points (xyu) in particle_filter.f90\n')
fout.write('%9i'%len(names)+' ! number of parameters to be identified\n')
fout.close()

# co-variance matrix for PF.exe
cov = input("assume coveriance:")
fout = file('co-variance_matrix.txt','w')
covMatrix = np.zeros([nObs,nObs])
for i in xrange(nObs):
	covMatrix[i,i] = cov
	fout.write(' '.join(['%10.6f'%covMatrix[i,j] for j in xrange(nObs)]+['\n']))
fout.close()

# measurement matrix for PF.exe
fout = file('obs_matrix.txt','w')
obsMatrix = np.identity(nObs)
for i in xrange(nObs):
	fout.write(' '.join(['%1i'%obsMatrix[i,j] for j in xrange(nObs)]+['\n']))
fout.close()

######################################
##  run the particle filter PF.exe  ##
######################################

# compile particle_filter.f90 is no executable is found
if not path.isfile("../PF.exe"):
	system("gfortran -o ../PF.exe ../particle_filter.f90")
# run the particle filter executable
system("../PF.exe")

##################################
##  plot identified parameters  ##
##################################

ips = np.genfromtxt('IP.txt')
weight = np.genfromtxt('weight.txt')
params = np.genfromtxt('particle.txt')
# plot evolutions of identified parameters (weighted average) 
# and probability density function of identified parameters at selected steps
plotIPs(names,ips,nSample,weight,params)
