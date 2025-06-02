from Explanations_Models.Custom_DT.LIME import LIME
from FeatureImportance.FI import FeatureImportance
import yaml
import pandas as pd
import argparse
import os
import sys
import numpy as np
import torch
import gym
import matplotlib
import matplotlib.pyplot as plt
import unittest
from DeepLearning_Models.utils.general import join, plot_combined
from DeepLearning_Models.ActorCritic.policy_gradient import PolicyGradient
from EnvRunner import GymRunner
import json
import warnings
from tqdm import tqdm
import torch
# Suppress all deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

yaml.add_constructor("!join", join)
parser = argparse.ArgumentParser()

parser.add_argument("--config_env", required=True, type=str)
parser.add_argument("--norm", required=True, type=int)
parser.add_argument("--samps", required=True, type=int)
parser.add_argument("--num", required=True, type=int)
args = parser.parse_args()
domain = args.config_env
if args.norm > 0:
    normalize = False
else:
    normalize = True

samps=args.samps
num = args.num
config_file = open("config_envs/{}.yml".format(domain))
config = yaml.load(config_file, Loader=yaml.FullLoader)
config.update(yaml.load(open("config_explanations/{}.yml".format(domain), encoding="utf8"), Loader= yaml.FullLoader))
Runner = GymRunner(config)
Runner.load_weights(PATH = None)


config["FI"]["method"] = "LRP"
config["sampler"]["num_samples"] = samps
config["normalize_FI"] = normalize
FI_calc = FeatureImportance(config, Runner.model.network)
DTMode = LIME(config, Runner, FI_getta=FI_calc)
X,Y = DTMode.sample_set()
out,FI_in_LRP = DTMode.surr_model.model.get_FI(X)

config["FI"]["method"] = "FD"
config["sampler"]["num_samples"] = samps
config["normalize_FI"] = normalize
FI_calc = FeatureImportance(config, Runner.model.network)
DTMode = LIME(config, Runner, FI_getta=FI_calc)
out,FI_in_FD = DTMode.surr_model.model.get_FI(X)



X = pd.DataFrame(X, columns=config["picture"]["labels"])
if config["surrogate"]["classifier"]:
    Y = pd.DataFrame(Y, columns=["out"])
else:
    Y = pd.DataFrame(Y, columns=config["picture"]["class_names"])



dfs = [X, Y, FI_in_LRP, FI_in_FD, out]

# Folder path
folder_path = "output_folder/" + domain +"norm"+str(config["normalize_FI"])+ "/"

# Check if the folder exists, if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Save each DataFrame in the folder
names = ["X", "Y", "FI_LRP", "FI_FD", "out"]
for i, df in enumerate(dfs, 1):
    print(i)
    df.to_csv(f"{folder_path}/"+names[i-1]+str(num)+".csv", index=False)

print("DataFrames saved successfully!")