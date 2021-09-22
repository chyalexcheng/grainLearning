import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from os import system, listdir, path
from matplotlib.offsetbox import AnchoredText
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

params = {'lines.linewidth': 1.5,'backend': 'ps','axes.labelsize': 18, 'axes.titlesize': 18, 'font.size': 18, 'legend.fontsize': 10.5,'xtick.labelsize': 15,'ytick.labelsize': 15,'text.usetex': True,'font.family': 'serif','legend.edgecolor': 'k','legend.fancybox':False}
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams.update(params)

def plotSigAndESS(sigAndESS):
	fig = plt.figure(figsize=(18/2.54,12.5/2.54))
	sigAndESS = np.array(sigAndESS) 
	plt.plot(sigAndESS[:,0],sigAndESS[:,1],'k')
	plt.xlabel(r'$\sigma$',labelpad=12); plt.ylabel(r'$ESS$',labelpad=16)
	plt.xlim(0,sigAndESS[0,0])
	plt.grid(True); plt.tight_layout();
	plt.savefig('sigAndESS.pdf',dpi=600); plt.show()
