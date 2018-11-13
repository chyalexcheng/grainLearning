# README #

*Grain*Learning: a Bayesian calibration tool for discrete element simulations of granular materials

Given an initial guess of parameter values, parameter space is iteratively explored, with a multi-level sampling algorithm to target at parameter subspaces where the posterior probabilities are high. Sampled parameter values are passed to the open-source DEM package YADE (http://yade-dem.org/) for model evaluations. smc.py contains the class that bridge the simulation data and experimental data with Bayesian statistics. A python script should be provided by the user to run DEM simulations for specific applications. 

The following packages are needed:

* Primary:

    Discrete Element Method: Yade (http://yade-dem.org/)

    Halton number generator: ghalton (https://pypi.python.org/pypi/ghalton)

    Nonparametric density estimation: scikit-learn (https://scikit-learn.org)

* Others:

    numpy

    scipy

    matplotlib
 
Files

   *grainLearning/test.py is an example script which uses existing simulation and experimental data to infer the posterior distribution of micromechanical parameters, conditioned on the stress-strain behavior of glass beads.

   *grainLearning/smc.py contains the smc class. The sequential Monte Carlo filter is implemented here to recursively approximate the posterior distribution of model parameters. It has a member function resampleParams which calls the BayesianGaussianMixture model in scikit-learn to resample parameter space in a nonparametric manner.

   *grainLearning/tools.py contains the functions that does the training of the BayesianGaussianMixture model. Functions that write parameter values in a text file for running YADE in the batch mode are included as well.

   *gmm_iterPF0-3.pkl are the trained BayesianGaussianMixture models
