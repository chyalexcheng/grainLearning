""" Author: Hongyang Cheng <chyalexcheng@gmail>
	 An interface to generate parameter table with Halton sequence
	 (requires ghalton library: https://pypi.python.org/pypi/ghalton)
"""

from math import *
import ghalton
import numpy as np

def paramsTable(keys,maxs,mins,num=100,thread=4):
	"""
   :param dim: type integer, number of parameters
   :param num: type integer, number of sampling points for Monte Carlo Simulation
   :param thread: type integer, number of thread for each parallel simulation
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

	# output parameter table with thread number for each Yade simulation session
	tableName = 'smcTable.txt'
	fout = file(tableName,'w')
	fout.write(' '.join(['!OMP_NUM_THREADS','key']+keys+['\n']))
	for j in xrange(num):
		fout.write(' '.join(['%2i'%thread,'%9i'%j]+['%15.5e'%table[j][i] for i in xrange(dim)]+['\n']))
	fout.close()
	return table, tableName

def getKeysAndData(fileName):
	data = np.genfromtxt(fileName)
	fopen = open(fileName,'r')
	keys = (fopen.read().splitlines()[0]).split('\t\t')
	if '#' in keys: keys.remove('#')
	keysAndData = {}
	for key in keys: keysAndData[key] = data[:,keys.index(key)]
	return keysAndData
