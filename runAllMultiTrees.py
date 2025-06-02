from Explanations_Models.Custom_DT.LIME import LIME
from FeatureImportance.FI import FeatureImportance
import yaml
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

# Suppress all deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

yaml.add_constructor("!join", join)
parser = argparse.ArgumentParser()

#parser.add_argument("--samps", required=True, type=int)
parser.add_argument("--config_env", required=True, type=str)
parser.add_argument("--norm", required=True, type=int)

args = parser.parse_args()
import gc
import copy
config_env = args.config_env
config_file = open("config_envs/{}.yml".format(config_env))
config = yaml.load(config_file, Loader=yaml.FullLoader)
config.update(yaml.load(open("config_explanations/{}.yml".format(config_env), encoding="utf8"), Loader= yaml.FullLoader))
Runner = GymRunner(config)
Runner.load_weights(PATH = None)
Metrics = {}
#samps = args.samps
NumEpochs = 100
if args.norm == 1:
    config["normalize_FI"] = True
    fileLoc = "DTMetsMultiNormd"
else:
    config["normalize_FI"] = False
    fileLoc = "DTMetsMulti"
fileLoc = fileLoc + "/" +config_env + "/"
os.makedirs(fileLoc, exist_ok=True)
import sys
sys.setrecursionlimit(2000)
for samps in [3,5,10,20,40,60,100]:
    print("running")
    print(samps)
    config["surrogate"]["multi_tree"] = True
    for ep in tqdm(range(NumEpochs)):
        config["surrogate"]["use_FI"] = True
        config["sampler"]["num_samples"] = samps
        config["FI"]["grouping"] = "Max_avg"
        
        FI = FeatureImportance(config, Runner.model.network)
        DTMode = LIME(config, Runner, FI_getta=FI)
        
        #DTMod = LIME(config, Runner)
        X,Y = DTMode.sample_set()
        out,FI_in = DTMode.surr_model.model.get_FI(X)
        config["surrogate"]["use_FI"] = False
        config["sampler"]["num_samples"] = samps
        config["FI"]["grouping"] = "Baseline"
        DTMode = LIME(config, Runner)
       
        DTMode.surr_model.fit(X,Y, FI_in=FI_in, out_logits=out)
        # DTMode.percent_Correct()
        # DTMode.uniform_Correct()
        mets = DTMode.get_metrics()
        
        with open(fileLoc + config["FI"]["grouping"] + str(samps), "a") as file:
            jstring = json.dumps(mets)
            file.write(jstring + '\n')
        DTMode.surr_model.Save(FilenameEnder="multiTree" + fileLoc+config["FI"]["grouping"] + str(samps)+".json")
        #DTMode.surr_model.model.delete_tree()
        #del DTMode
        #del FI
        #DTMode = None
        

        #print(config["FI"]["grouping"])
        config["surrogate"]["use_FI"] = True
        config["sampler"]["num_samples"] = samps
        config["FI"]["grouping"] = "Max_avg"
        FI = FeatureImportance(config, Runner.model.network)
        DTMode = LIME(config, Runner,FI_getta=FI)
      
        DTMode.surr_model.fit(X,Y, FI_in=FI_in, out_logits=out)
        #DTMode.percent_Correct()
        #DTMode.uniform_Correct()
        mets = DTMode.get_metrics()
        with open(fileLoc + config["FI"]["grouping"] + str(samps), "a") as file:
            jstring = json.dumps(mets)
            file.write(jstring + '\n')
        DTMode.surr_model.Save(FilenameEnder="multiTree" + fileLoc+config["FI"]["grouping"] + str(samps)+".json")
