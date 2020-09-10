
# Welcome to GrainLearning!

|Author |Hongyang Cheng|
|---|---|
|Contacts |h.cheng@utwente.nl |
|Version|0.1 |

# Description

Note that GrainLearning 0.1 is a fortran code where the sequential Monte Carlo algorithm (not yet iteratively) is implemented for the **Bayesian Calibration** of discrete element models.
It uses the [recursive Bayes' rule](https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation)
to quantify the evolution of the probability distribution of parameters over time or a load history.
In this version, samples are drawn only uniformly, assuming no prior knowledge.
Therefore, the efficiency is not optimal. However, one may find the fortran code useful and easy to understand than GrainLearning 0.2 in the directory ../grainLearning.
In this directory, you can find the source code for the sequential Monte Carlo filtering and several Yade scripts for running triaxial compression simulations.
We also share with you several dense granular packings we prepared that have an identical porosity but different numbers of particles.

# Dependencies

* Discrete Element Method: Yade (http://yade-dem.org/)
* Halton number generator: ghalton (https://pypi.python.org/pypi/ghalton)
* Nonparametric Gaussian mixture model: scikit-learn (https://scikit-learn.org)
* multiprocessing, numpy, scipy, matplotlib, etc.

# Referencing GrainLearning

If you are using this software in a work that will be published, please cite this paper:

H. Cheng, T. Shuku, K. Thoeni, Y. Yamamoto. **Probabilistic calibration of discrete element simulations using the sequential quasi-Monte Carlo filter.** _Granular Matter_ 20, 11 (2018). [10.1007/s10035-017-0781-y](https://doi.org/10.1007/s10035-017-0781-y)

H. Cheng, T. Shuku, K. Thoeni, P. Tempone, S. Luding, V. Magnanimo. **An iterative Bayesian filtering framework for fast and automated calibration of DEM models**. _Comput. Methods Appl. Mech. Eng.,_ 350 (2019), pp. 268-294, [10.1016/j.cma.2019.01.027](https://doi.org/10.1016/j.cma.2019.01.027)

# Help and Support

For assistance with the GrainLearning software or Bayesian calibration for geomechanical models in general, please raise an issue on the Github Issues page or drop me an email at h.cheng@utwente.nl.
