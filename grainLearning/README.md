# README #

GrainLearning: a Bayesian calibration tool for discrete element simulations of granular materials

Given an initial guess of parameter values, parameter space is iteratively explored, with a multi-level sampling algorithm to target at parameter subspaces where the posterior probabilities are high. Sampled parameter values are passed to the open-source DEM package YADE (http://yade-dem.org/) for model evaluations. smc.py contains the class that bridge the simulation data and experimental data with Bayesian statistics. A python script should be provided by the user to run DEM simulations for specific applications. Make sure the simulation data file is named as 'simName_key_irrelevantParams_unKnownParam0_unKnownParam1_..unKnownParamN 

The following packages are needed:

* Primary:

    Discrete Element Method: Yade (http://yade-dem.org/)

    Halton number generator: ghalton (https://pypi.python.org/pypi/ghalton)

    Nonparametric density estimation: scikit-learn (https://scikit-learn.org)

* Others:

    numpy, scipy, matplotlib
 
Examples

  * **Integrated mode (standAlone=False):**
   *GrainLearning/CaliCollision.py* does Bayesian simulation with DEM simulations (YADE) running at the same time within a Python environment. 
   
  * **Standalone mode with existing simulation data (standAlone=False, skipDEM=True):** *GrainLearning/CaliOedoCompress.py* uses pre-run DEM simulation data and experimental results to infer the posterior distribution of micromechanical parameters, conditioned on the stress-strain behavior of glass beads (see https://www.sciencedirect.com/science/article/pii/S0045782519300520 for further details).

  * **Sequential Monte Carlo (SMC)**: *GrainLearning/smc.py* contains the smc class. The sequential Monte Carlo filter is implemented here to recursively approximate the posterior distribution of model parameters. It has a member function resampleParams which calls the BayesianGaussianMixture model in scikit-learn to resample parameter space in a nonparametric manner.

  * GrainLearning/tools.py contains the functions that does the data-driven **training** of the Bayesian Gaussian Mixture model. The trained statistical model resamples parameter space at the end of each iteration. For example, gmm_iterPF0-3.pkl are the Gaussian mixture models trained with data in iterPF0-3, saved in python pickle format. Functions such as writing parameter values in a text file for running YADE in the batch mode are also included here. 

  * For **single-iteration Bayesian calibration**, users could try modifying a much simplified driver code *GrainLearning/CaliTemplate.py* where all the flags and variables are explained.

Note one can also apply GrainLearning for Bayesian calibration of other DEM codes, by passing parameter samples to the executables via a shell or python script (GrainLearning/smc.py: lines 125-129). 