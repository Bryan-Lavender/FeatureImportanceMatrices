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

import random
import yaml

yaml.add_constructor("!join", join)
parser = argparse.ArgumentParser()
parser.add_argument("--config_filename", required=False, type=str)
parser.add_argument("--plot_config_filename", required=False, type=str)
parser.add_argument("--run_basic_tests", required=False, type=bool)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.config_filename is not None:
        config_file = open("config_envs/{}.yml".format(args.config_filename))
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        

        for seed in config["env"]["seed"]:
            torch.random.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            print(config["env"]["env_name"])
            env = gym.make(config["env"]["env_name"])

            # train model
            model = PolicyGradient(env, config, seed)
            model.run()
    
    else:
        print("no instruction")
