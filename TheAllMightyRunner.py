import argparse
import os
import sys
import numpy as np
import torch
import gym
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import unittest
from DeepLearning_Models.utils.general import join, plot_combined
from DeepLearning_Models.ActorCritic.policy_gradient import PolicyGradient
from EnvRunner import GymRunner
import random
import yaml
yaml.add_constructor("!join", join)
parser = argparse.ArgumentParser()

parser.add_argument("--config_filename", required=False, type=str)
parser.add_argument("--task", required=True, type=str)

def weight_dict_corrector(loaded_state_dict):
    new_state_dict = {}
    for key, value in loaded_state_dict.items():
        new_key = key.replace("network.", "")  # Remove the 'network.' prefix
        new_state_dict[new_key] = value
    return new_state_dict

"""
init train
    args: model (PolicyGradient)
    fits weights to RL environment
"""
def train(model: PolicyGradient):
    model.train()


"""
init record
    args: runner (GymRunner)
    records an execution of the policy
"""
def record(runner: GymRunner):
    runner.recorder()

"""
init run_with_record_comparison
    args: model (PolicyGradient)
    records first and last execution of the model when training
"""
def run_with_record_comparison(model: PolicyGradient):
    model.run()

if __name__ == "__main__":
    args = parser.parse_args()
    config_file = open("config_envs/{}.yml".format(args.config_filename))
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    env = gym.make(config["env"]["env_name"])
    seed = config["env"]["seed"]
    
    runner_thang = GymRunner(config, env)
    model = PolicyGradient(runner_thang.runner, runner_thang.recorder, config, seed)
    runner_thang.init_model(model)
    if args.task == "train":
        run_with_record_comparison(model)
    elif args.task == "run":
        record(runner_thang)
    