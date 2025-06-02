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
from Explanations_Models.LIME import LimeModel

import random
import yaml
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
    config_file = open("config_explanations/{}.yml".format("CartPole_Gaussian_All"))
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    env1 = gym.make(config["env"]["env_name"], render_mode="rgb_array")
    env2 = gym.make(config["env"]["env_name"], render_mode="rgb_array")
    seed = config["env"]["seed"][0]
    model = PolicyGradient(env1, config, seed)
    
    model.network.load_state_dict(weight_dict_corrector(torch.load(config["output"]["actor_output"].format(seed))))
    model.baseline_network.load_state_dict(torch.load(config["output"]["critic_output"].format(seed)))
    LimeModel = LimeModel(model.network, torch.tensor([0.,0.,0.,0.]), config)
    t = 0
    for i in LimeModel.interpretable_models:
        if t == 0:
            i.load_state_dict(torch.load("SurrigateWeights/model1/0Dec.pt"))
        else:
            i.load_state_dict(torch.load("SurrigateWeights/model1/1Dec.pt"))
        t += 1
    

    print("Weights loaded")
    state_diffs = []
    action_diffs = []
    reward_diffs = []
    MainPolicyActionDiff=[]
    for i_episode in range(1):  # Run for a few episodes
        state1, info1 = env1.reset()
        env1 = gym.wrappers.RecordVideo(
            env1,
            config["explanation_output"]["policy_video"],
            step_trigger=lambda x: x % 100 == 0,
        )
        rewards1 = []

        state2, info2 = env2.reset()
        env2 = gym.wrappers.RecordVideo(
            env2,
            config["explanation_output"]["exp_video"],
            step_trigger=lambda x: x % 100 == 0,
        )
        rewards2 = []
        term_1 = True
        term_2 = True
        for t in range(1000):  # Maximum number of steps per episode
            
            if term_1:
                action1 = model.policy.act(state1)
                state1, reward1, terminated1, truncated1, info1 = env1.step(action1)
                rewards1.append(reward1)
                if terminated1:
                    print("done, reward:", sum(rewards1))
                    print(rewards1)
                    term_1 = False
                    reward1 = 0
            if term_2:
                
                a0 = LimeModel.interpretable_models[0](torch.tensor(state2))
                a1 = LimeModel.interpretable_models[1](torch.tensor(state2))
                if a0 > a1:
                    action2 = 0
                else:
                    action2 = 1
                state2, reward2, terminated2, truncated2, info2 = env2.step(action2)
                rewards2.append(reward2)
                if terminated2:
                    print("done, reward:", sum(rewards2))
                    print(rewards2)
                    term_2 = False
                    reward2 = 0
            

            state_diffs.append(np.linalg.norm(state1-state2))
            action_diffs.append(np.linalg.norm(action1-action2))
            reward_diffs.append(np.linalg.norm(reward1-reward2))
            

            a0 = LimeModel.interpretable_models[0](torch.tensor(state1))
            a1 = LimeModel.interpretable_models[1](torch.tensor(state1))
            if a0 > a1:
                action2_test = 0
            else:
                action2_test = 1
            

            MainPolicyActionDiff.append(np.linalg.norm(action1-action2_test))
            if not (term_1 or term_2):
                break
            
        np.save(config["explanation_output"]["state_dist"],np.array(state_diffs))
        np.save(config["explanation_output"]["action_dist"],np.array(action_diffs))
        np.save(config["explanation_output"]["reward_dist"],np.array(reward_diffs))
        np.save(config["explanation_output"]["policy_dist"], np.array(MainPolicyActionDiff))

    env1.close()