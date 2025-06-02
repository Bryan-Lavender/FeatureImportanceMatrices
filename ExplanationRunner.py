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
from Explanations_Models.DT_LIME.LIME import LIME
import json
import warnings
from tqdm import tqdm

# Suppress all deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

yaml.add_constructor("!join", join)
parser = argparse.ArgumentParser()

parser.add_argument("--config_filename", required=False, type=str)
#parser.add_argument("--task", required=True, type=str)

class MetricGetter():
    def __init__(self, config, Runner):
        self.config = config
        self.Runner = Runner
    
    def run_series(self):
        config = self.config
        Runner = self.Runner
        PercentCorrect = []
        ExecutionMSE = []
        ExecutionDiff = []
        TDepth = []
        TBreadth = []
        top_splits = []
        UniformCorrect = []
        for i in tqdm(range(config["metric_hyperparameters"]["tree_execution_samples"])):
            limemod = LIME(config, Runner)
            limemod.train()
            PercentCorrect.append(float(limemod.percent_Correct()))
            UniformCorrect.append(float(limemod.uniform_Correct()))
            ADmse, CountDiff = limemod.absolute_distance()
            ExecutionMSE.append(float(ADmse))
            ExecutionDiff.append(CountDiff)

            depth, breadth = limemod.surr_model.depth_breadth()
            TDepth.append(depth)
            TBreadth.append(breadth)
            top_splits.append(config["picture"]["labels"][limemod.surr_model.get_top_split()])
            path = limemod.surr_model.Save(FilenameEnder="tree_"+str(i)+".pkl")
        return {"PercentCorrect": PercentCorrect, "UniformCorrect": UniformCorrect,"EpisodeDistance": ExecutionMSE, "Episode_Length_Distance": ExecutionDiff, "Depth": TDepth, "Breadth": TBreadth, "TopSplits": top_splits, "Path": path}

    def sample_rate(self):
        seq = self.config["metric_hyperparameters"]["sample_sequence"]
        returner = {}
        for i in seq:
            self.config["sampler"]["num_samples"] = i
            print("sample_num:", i)
            out = self.run_series()
            returner[str(i)] = out
        
        return returner

    def Saver(self, metrics, save_img = False):
        path = self.config["exp_output"]["output_path"]
        path = path + "/" + self.config["surrogate"]["criterion"] + ".json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as json_file:
            json.dump(metrics, json_file)
        return path
        
    
    def run_sample_rates(self):
        returns = self.sample_rate()
        self.Saver(returns)
    
    def run_samples_with_types(self):
        for i in self.config["metric_hyperparameters"]["citerions"]:
            self.config["surrogate"]["criterion"] = i
            print("running", i)
            self.run_sample_rates()
            print("saved", i)

    
        
if __name__ == "__main__":
    args = parser.parse_args()
    config_file = open("config_envs/{}.yml".format(args.config_filename))
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    config.update(yaml.load(open("config_explanations/{}.yml".format(args.config_filename), encoding="utf8"), Loader= yaml.FullLoader))
    Runner = GymRunner(config)
    Runner.load_weights(PATH = None)
    MG = MetricGetter(config, Runner)
    MG.run_samples_with_types()

