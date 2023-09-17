"""
This tutorial shows how to perform iterative Bayesian calibration for a linear regression model
 using GrainLearning.
"""
import os
from math import floor, log
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem

PATH = os.path.abspath(os.path.dirname(__file__))
executable = f'python {PATH}/linear_model.py'


def run_sim(system):
    """
    Run the external executable and passes the parameter sample to generate the output file.
    """
    # keep the naming convention consistent between iterations
    mag = floor(log(system.num_samples, 10)) + 1
    # check the software name and version
    print("*** Running external software... ***\n")
    # loop over and pass parameter samples to the executable
    for i, params in enumerate(system.param_data):
        description = 'Iter' + str(system.curr_iter) + '_Sample' + str(i).zfill(mag)
        print(" ".join([executable, "%.8e %.8e" % tuple(params), system.sim_name, description]))
        os.system(' '.join([executable, "%.8e %.8e" % tuple(params), system.sim_name, description]))


calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 10,
        "system": {
            "system_type": IODynamicSystem,
            "param_min": [0.001, 0.001],
            "param_max": [1, 10],
            "param_names": ['a', 'b'],
            "num_samples": 20,
            "obs_data_file": PATH + '/linear_obs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'linear',
            "sim_data_dir": PATH + '/sim_data/',
            "sim_data_file_ext": '.txt',
            "sigma_tol": 0.01,
            "callback": run_sim,
        },
        "calibration": {
            "inference": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 2,
                "n_init": 1,
                "random_state": 0,
                "covariance_type": "full",
            }
        },
        "save_fig": 0,
    }
)

calibration.run()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')

error_tolerance = 0.1

error = most_prob_params - [0.2, 5.0]
assert abs(
    error[0]) / 0.2 < error_tolerance, f"Model parameters are not correct, expected 0.2 but got {most_prob_params[0]}"
assert abs(
    error[1]) / 5.0 < error_tolerance, f"Model parameters are not correct, expected 5.0 but got {most_prob_params[1]}"
