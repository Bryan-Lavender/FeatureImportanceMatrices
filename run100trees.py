
# Folder path where CSV files are stored
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
import random
# Suppress all deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

yaml.add_constructor("!join", join)
parser = argparse.ArgumentParser()

parser.add_argument("--config_filename", required=False, type=str)
parser.add_argument("--max_number", required=False, type=int)
args = parser.parse_args()
domain = args.config_filename
normalize = True

config_file = open("config_envs/{}.yml".format(domain))
config = yaml.load(config_file, Loader=yaml.FullLoader)
config.update(yaml.load(open("config_explanations/{}.yml".format(domain), encoding="utf8"), Loader= yaml.FullLoader))
Runner = GymRunner(config)
Runner.load_weights(PATH = None)

def get_dataframes(domain, max_number):
    normalize = True
    folder_path = "output_folder_tmp/" + domain +"norm"+str(normalize)
    # File prefixes for each set of files
    def read_multi_headed_csv(file_path):
        return pd.read_csv(file_path, header=[0, 1])  # This assumes a 2-level header

    # Concatenate files for multi-index columns like FI_FD and FI_LRP
    file_types = ['X', 'Y', 'FI_FD', 'FI_LRP', 'out']

    dataframes = {}
    for file_type in file_types:
        file_list = [f"{folder_path}/{file_type}{i}.csv" for i in range(1, max_number+1)]  # Adjust range as needed

        # Use read_multi_headed_csv for FI_FD and FI_LRP
        if file_type in ['FI_FD', 'FI_LRP']:
            df_list = [read_multi_headed_csv(file) for file in file_list]
        else:
            df_list = [pd.read_csv(file) for file in file_list]

        # Concatenate them, preserving the multi-index headers
        dataframes[file_type] = pd.concat(df_list, ignore_index=True)
    return dataframes


max_number = args.max_number
dataframes = get_dataframes(domain, max_number)
fileLoc = "LRPCorrectly/"+domain+"/metrics/"
if not os.path.exists(fileLoc):
        os.makedirs(fileLoc)
fileLocTrees = "LRPCorrectly/"+domain+"/trees/"
if not os.path.exists(fileLocTrees):
        os.makedirs(fileLocTrees)
# Load the DataFrame from the CSV file
df_reloaded = pd.read_csv("sampled_indices.csv")

# Convert the DataFrame back to a list of lists (if necessary)
sampled_lists_reloaded = df_reloaded.values.tolist()

Sample_size = [3,5,10,20,40,60,100]
for samps in Sample_size:
    for i in tqdm(range(0,100)):
        unique_indices = np.array(sampled_lists_reloaded[i])[0:samps]
        

        config["surrogate"]["multi_tree"] = False
        config["FI"]["method"] = "FD"
        
        """
        Begin FD Single Tree
        """
        X = dataframes["X"].loc[unique_indices]
        Y = dataframes["Y"].loc[unique_indices]
        #FI_LRP = dataframes["FI_FD"].loc[unique_indices]
        FI_FD = dataframes["FI_LRP"].loc[unique_indices]
        out = dataframes["out"].loc[unique_indices]

        config["surrogate"]["use_FI"] = False
        config["sampler"]["num_samples"] = samps
        config["FI"]["grouping"] = "Baseline"
        DTMode = LIME(config, Runner)
        
        #DTMode.surr_model.fit(X,Y, FI_in=FI_in, out_logits=out)
        DTMode.surr_model.fit(X,Y)
        # DTMode.percent_Correct()
        # DTMode.uniform_Correct()
        mets = DTMode.get_metrics()
        
        with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size), "a") as file:
            jstring = json.dumps(mets)
            file.write(jstring + '\n')
        DTMode.surr_model.Save(FilenameEnder= fileLocTrees+str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size)+" "+str(i)+".json")

        


        config["surrogate"]["use_FI"] = True
        config["sampler"]["num_samples"] = samps
        config["FI"]["grouping"] = "Max_avg"
        DTMode = LIME(config, Runner)
        
        DTMode.surr_model.fit(X,Y, FI_in=FI_FD, out_logits=out)
        #DTMode.surr_model.fit(X,Y)
        # DTMode.percent_Correct()
        # DTMode.uniform_Correct()
        mets = DTMode.get_metrics()
        with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size), "a") as file:
            jstring = json.dumps(mets)
            file.write(jstring + '\n')
        DTMode.surr_model.Save(FilenameEnder= fileLocTrees+str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size)+" "+str(i)+".json")



        config["surrogate"]["use_FI"] = True
        config["sampler"]["num_samples"] = samps
        config["FI"]["grouping"] = "Max_all"
        DTMode = LIME(config, Runner)
        
        DTMode.surr_model.fit(X,Y, FI_in=FI_FD, out_logits=out)
        #DTMode.surr_model.fit(X,Y)
        # DTMode.percent_Correct()
        # DTMode.uniform_Correct()
        mets = DTMode.get_metrics()
        
        with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size), "a") as file:
            jstring = json.dumps(mets)
            file.write(jstring + '\n')
        DTMode.surr_model.Save(FilenameEnder= fileLocTrees+str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size)+" "+str(i)+".json")


        config["surrogate"]["use_FI"] = True
        config["sampler"]["num_samples"] = samps
        config["FI"]["grouping"] = "Var_weighted"
        DTMode = LIME(config, Runner)
        
        DTMode.surr_model.fit(X,Y, FI_in=FI_FD, out_logits=out)
        #DTMode.surr_model.fit(X,Y)
        # DTMode.percent_Correct()
        # DTMode.uniform_Correct()
        mets = DTMode.get_metrics()
        
        with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size), "a") as file:
            jstring = json.dumps(mets)
            file.write(jstring + '\n')
        DTMode.surr_model.Save(FilenameEnder= fileLocTrees+str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size)+" "+str(i)+".json")

        """
        Begin FD Multi_Tree
        """
    
        config["surrogate"]["multi_tree"] = True
        config["surrogate"]["use_FI"] = False
        config["sampler"]["num_samples"] = samps
        config["FI"]["grouping"] = "Baseline"
        DTMode = LIME(config, Runner)
        
        DTMode.surr_model.fit(X,Y, FI_in=FI_FD, out_logits=out)
        #DTMode.surr_model.fit(X,Y)
        # DTMode.percent_Correct()
        # DTMode.uniform_Correct()
        mets = DTMode.get_metrics()
        
        with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size), "a") as file:
            jstring = json.dumps(mets)
            file.write(jstring + '\n')
        DTMode.surr_model.Save(FilenameEnder= fileLocTrees+str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size)+" "+str(i)+".json")

        config["surrogate"]["multi_tree"] = True
        config["surrogate"]["use_FI"] = True
        config["sampler"]["num_samples"] = samps
        config["FI"]["grouping"] = "Max_avg"
        DTMode = LIME(config, Runner)
        
        DTMode.surr_model.fit(X,Y, FI_in=FI_FD, out_logits=out)
        #DTMode.surr_model.fit(X,Y)
        # DTMode.percent_Correct()
        # DTMode.uniform_Correct()
        mets = DTMode.get_metrics()
        
        with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size), "a") as file:
            jstring = json.dumps(mets)
            file.write(jstring + '\n')
        DTMode.surr_model.Save(FilenameEnder= fileLocTrees+str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + "LRPSamps" + str(Sample_size)+" "+str(i)+".json")



    # config["surrogate"]["multi_tree"] = False
    # config["FI"]["method"] = "LRP"
    
    # """
    # Begin LRP Single Tree
    # """
  
    # config["surrogate"]["use_FI"] = False
    # config["sampler"]["num_samples"] = Sample_size
    # config["FI"]["grouping"] = "Baseline"
    # DTMode = LIME(config, Runner)
    
    # #DTMode.surr_model.fit(X,Y, FI_in=FI_in, out_logits=out)
    # DTMode.surr_model.fit(X,Y)
    # # DTMode.percent_Correct()
    # # DTMode.uniform_Correct()
    # mets = DTMode.get_metrics()
    
    # with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + config["FI"]["method"], "a") as file:
    #     jstring = json.dumps(mets)
    #     file.write(jstring + '\n')
    # DTMode.surr_model.Save(FilenameEnder= fileLocTrees + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"]+  config["FI"]["method"]+str(i)+".json")

    


    # config["surrogate"]["use_FI"] = True
    # config["sampler"]["num_samples"] = Sample_size
    # config["FI"]["grouping"] = "Max_avg"
    # DTMode = LIME(config, Runner)
    
    # DTMode.surr_model.fit(X,Y, FI_in=FI_LRP, out_logits=out)
    # #DTMode.surr_model.fit(X,Y)
    # # DTMode.percent_Correct()
    # # DTMode.uniform_Correct()
    # mets = DTMode.get_metrics()
    
    # with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + config["FI"]["method"], "a") as file:
    #     jstring = json.dumps(mets)
    #     file.write(jstring + '\n')
    # DTMode.surr_model.Save(FilenameEnder= fileLocTrees + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"]+  config["FI"]["method"]+str(i)+".json")



    # config["surrogate"]["use_FI"] = True
    # config["sampler"]["num_samples"] = Sample_size
    # config["FI"]["grouping"] = "Max_all"
    # DTMode = LIME(config, Runner)
    
    # DTMode.surr_model.fit(X,Y, FI_in=FI_LRP, out_logits=out)
    # #DTMode.surr_model.fit(X,Y)
    # # DTMode.percent_Correct()
    # # DTMode.uniform_Correct()
    # mets = DTMode.get_metrics()
    
    # with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + config["FI"]["method"], "a") as file:
    #     jstring = json.dumps(mets)
    #     file.write(jstring + '\n')
    # DTMode.surr_model.Save(FilenameEnder= fileLocTrees + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"]+  config["FI"]["method"]+str(i)+".json")



    # config["surrogate"]["use_FI"] = True
    # config["sampler"]["num_samples"] = Sample_size
    # config["FI"]["grouping"] = "Var_weighted"
    # DTMode = LIME(config, Runner)
    
    # DTMode.surr_model.fit(X,Y, FI_in=FI_LRP, out_logits=out)
    # #DTMode.surr_model.fit(X,Y)
    # # DTMode.percent_Correct()
    # # DTMode.uniform_Correct()
    # mets = DTMode.get_metrics()
    
    # with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + config["FI"]["method"], "a") as file:
    #     jstring = json.dumps(mets)
    #     file.write(jstring + '\n')
    # DTMode.surr_model.Save(FilenameEnder= fileLocTrees + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"]+  config["FI"]["method"]+str(i)+".json")


    # """
    # Begin FD Multi_Tree
    # """
    # config["surrogate"]["multi_tree"] = True
    # config["surrogate"]["use_FI"] = False
    # config["sampler"]["num_samples"] = Sample_size
    # config["FI"]["grouping"] = "Baseline"
    # DTMode = LIME(config, Runner)
    
    # DTMode.surr_model.fit(X,Y, FI_in=FI_FD, out_logits=out)
    # #DTMode.surr_model.fit(X,Y)
    # # DTMode.percent_Correct()
    # # DTMode.uniform_Correct()
    # mets = DTMode.get_metrics()
    
    # with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + config["FI"]["method"], "a") as file:
    #     jstring = json.dumps(mets)
    #     file.write(jstring + '\n')
    # DTMode.surr_model.Save(FilenameEnder= fileLocTrees + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"]+  config["FI"]["method"]+str(i)+".json")


    # config["surrogate"]["multi_tree"] = True
    # config["surrogate"]["use_FI"] = True
    # config["sampler"]["num_samples"] = Sample_size
    # config["FI"]["grouping"] = "Max_avg"
    # DTMode = LIME(config, Runner)
    
    # DTMode.surr_model.fit(X,Y, FI_in=FI_LRP, out_logits=out)
    # #DTMode.surr_model.fit(X,Y)
    # # DTMode.percent_Correct()
    # # DTMode.uniform_Correct()
    # mets = DTMode.get_metrics()
    
    # with open(fileLoc + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"] + config["FI"]["method"], "a") as file:
    #     jstring = json.dumps(mets)
    #     file.write(jstring + '\n')
    # DTMode.surr_model.Save(FilenameEnder= fileLocTrees + str(config["surrogate"]["multi_tree"]) + config["FI"]["grouping"]+  config["FI"]["method"]+str(i)+".json")



    


