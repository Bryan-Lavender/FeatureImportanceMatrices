import argparse
import os
import sys
import numpy as np
import torch
import gym
import matplotlib
import time
matplotlib.use("agg")
import matplotlib.pyplot as plt
import unittest
from DeepLearning_Models.utils.general import join, plot_combined
from DeepLearning_Models.ActorCritic.policy_gradient import PolicyGradient
from Explanations_Models.LIME import LimeModel
import random
import yaml
from array import array

yaml.add_constructor("!join", join)
parser = argparse.ArgumentParser()


def select_action(state, actor_model):
    # Assuming your actor model directly outputs the action to take
    # You might need to process the output depending on your model architecture
    with torch.no_grad():
        state = torch.from_numpy(state).float()
        action = actor_model(state).max(0)[1].view(1, 1)
    return action.item()

def weight_dict_corrector(loaded_state_dict):
    new_state_dict = {}
    for key, value in loaded_state_dict.items():
        new_key = key.replace("network.", "")  # Remove the 'network.' prefix
        new_state_dict[new_key] = value
    return new_state_dict

parser.add_argument("--config_filename", required=False, type=str)
parser.add_argument("--plot_config_filename", required=False, type=str)
parser.add_argument("--run_basic_tests", required=False, type=bool)


if __name__ == "__main__":
    args = parser.parse_args()
    config_file = open("config_explanations/{}.yml".format(args.config_filename))
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    device = config["model_training"]["device"]
    
    new_folder_path = config["explanation_output"]["save_path"]
    os.makedirs(new_folder_path, exist_ok=True)
    WeightSaves = config["explanation_weights"]["model_path"]
    os.makedirs(WeightSaves,exist_ok=True )
    seed = config["env"]["seed"][0]
    env1 = gym.make(config["env"]["env_name"], render_mode="rgb_array")
    
    model = PolicyGradient(env1, config, seed)
    
    model.network.load_state_dict(weight_dict_corrector(torch.load(config["output"]["actor_output"].format(seed))))
    model.baseline_network.load_state_dict(torch.load(config["output"]["critic_output"].format(seed)))
    point = np.array(array("f", config["sampling"]["point"]))
    train_time = []
    tmpstr_modelPath = config["explanation_weights"]["model_path"]
    tmpstr_MAE = config["explanation_output"]["MAE_MSE_RMSE_Rsq"]
    for i_episode in range(5):  # Run for a few episodes
        
        config["explanation_weights"]["model_path"] = tmpstr_modelPath + str(i_episode)
        config["explanation_output"]["MAE_MSE_RMSE_Rsq"] = tmpstr_MAE + str(i_episode)
        start = time.time()
        PolicyLimeModel = LimeModel(model.network, point, config)
        end = time.time()
        train_time.append(end - start)
        PolicyLimeModel.runner()
        print("model trained")
        state_diffs_dist = []
        state_diffs_vec = []
        MainPolicyActionDiff=[]
        avg_policy_correctness = []
        avg_uniform_correctness = []

        env1 = gym.make(config["env"]["env_name"], render_mode="rgb_array")
        env2 = gym.make(config["env"]["env_name"], render_mode="rgb_array")
        initial_state = np.random.uniform(0, 2, size=(4,))

        env1.reset()
        env1.env.state = initial_state
        state1 = initial_state
        state2 = initial_state
        
        env1 = gym.wrappers.RecordVideo(
            env1,
            config["explanation_output"]["policy_video"] + str(i_episode),
            step_trigger=lambda x: x % 100 == 0,
        )
        rewards1 = []
        env2.reset()
        env2.env.state = initial_state
        env2 = gym.wrappers.RecordVideo(
            env2,
            config["explanation_output"]["exp_video"] + str(i_episode),
            step_trigger=lambda x: x % 100 == 0,
        )
        rewards2 = []
        term_1 = True
        term_2 = True
        print("state1: ",state1)
        print("state2: ",state2)
        for t in range(1000):  # Maximum number of steps per episode
            
            if term_1:
                action1 = model.policy.act(state1)
                decisions_sur = [i(torch.tensor(state1, dtype=torch.float32, device=device)).item() for i in PolicyLimeModel.interpretable_models]
                action2_test = np.argmax(decisions_sur)
                state1, reward1, terminated1, truncated1, info1 = env1.step(action1)
                rewards1.append(reward1)
                if terminated1:
                    term_1 = False
                    reward1 = 0
            if term_2:
                decisions = [i(torch.tensor(state2, dtype=torch.float32, device=device)).item() for i in PolicyLimeModel.interpretable_models]
                action2 = np.argmax(decisions)
                state2, reward2, terminated2, truncated2, info2 = env2.step(action2)
                rewards2.append(reward2)
                if terminated2:
                    term_2 = False
                    reward2 = 0
            

            state_diffs_dist.append(np.linalg.norm(state1-state2))
            state_diffs_vec.append(np.abs(state1-state2))

            
           
            
            if term_1:
                MainPolicyActionDiff.append(np.abs(action1-action2_test))
                if action1 == action2_test:
                    avg_policy_correctness.append(1)
                else:
                    avg_policy_correctness.append(0)
                
            if not (term_1 or term_2):
                break
        
        #do uniform test metric
        testfile = np.load("state_list.npy")
        correct = np.load("decision_samples.npy")

        for i, correct in zip(testfile, correct):
            decisions = [i(torch.tensor(state2, dtype=torch.float32, device=device)).item() for i in PolicyLimeModel.interpretable_models]
            action2 = np.argmax(decisions)
            if correct == action2:
                    avg_uniform_correctness.append(1)
            else:
                avg_uniform_correctness.append(0)
        print("AVG Policy", np.average(avg_policy_correctness))
        print("AVG Uniform", np.average(avg_uniform_correctness))
        np.save(config["explanation_output"]["uniform_sample_perc"] + str(i_episode), np.array(avg_uniform_correctness))
        np.save(config["explanation_output"]["state_dist"] + str(i_episode),np.array(state_diffs_dist))
        np.save(config["explanation_output"]["state_dist_vec"] + str(i_episode),np.array(state_diffs_vec))

        np.save(config["explanation_output"]["policy_dist"]+ str(i_episode), np.array(MainPolicyActionDiff))
        np.save(config["explanation_output"]["policy_perc"]+ str(i_episode), np.array(avg_policy_correctness))
    np.save(config["explanation_output"]["time_saver"], np.array(train_time))
    print("complete")
