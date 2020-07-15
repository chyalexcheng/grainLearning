""" Author: Hongyang Cheng <chyalexcheng@gmail>
	 An interface to generate parameter table with Halton sequence
	 (requires ghalton library: https://pypi.python.org/pypi/ghalton)
"""

from math import *
import ghalton

def paramsTable(keys,maxs,mins,num=2000,thread=4,large=1e5):
	"""
   :param dim: type integer, number of parameters
   :param num: type integer, number of sampling points for Monte Carlo Simulation
   :param thread: type integer, number of thread for each parallel simulation
   :param maxs: type tuples, maximums ranges of parameters
   :param mins: type tuples, minimums ranges of parameters
   :param keys: type strings, names of parameters
   :param large: type float, generate halton number on the powers if above this value
	"""
	dim = len(keys)
	sequencer = ghalton.Halton(dim)
	table = sequencer.get(num)
	for i in range(dim):
		for j in range(num):
			# for other parameters of small values
			if mins[i] < large:
				mean = .5*(maxs[i]+mins[i])
				std  = .5*(maxs[i]-mins[i])
				table[j][i] = mean+(table[j][i]-.5)*2*std
			# for parameters of large values like Young's modulus, use halton numbers on the powers
			if mins[i] >= large:
				powMax = log10(maxs[i])
				powMin = log10(mins[i])
				meanPow = .5*(powMax+powMin)
				stdPow  = .5*(powMax-powMin)
				power = meanPow+(table[j][i]-.5)*2*stdPow
				table[j][i] = 10**power
	# output parameter table with thread number for each Yade simulation session
	fout = open('table.dat','w')
	fout.write(' '.join(['!OMP_NUM_THREADS','key']+keys+['\n']))
	for j in range(num):
		fout.write(' '.join(['%2i'%thread,'%9i'%j]+['%15.5e'%table[j][i] for i in range(dim)]+['\n']))
	fout.close()

	# prepare parameter table for the particle-filter calibration
	fout = open('particle.txt','w')
	for j in range(num):
		for i in range(dim):
			fout.write('%15.5e'%table[j][i])
		fout.write('\n')
	fout.close()
