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
    config_file = open("config_envs/{}.yml".format(args.config_filename))
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    env = gym.make(config["env"]["env_name"], render_mode="rgb_array")
    seed = config["env"]["seed"][0]
    model = PolicyGradient(env, config, seed)
    
    model.network.load_state_dict(weight_dict_corrector(torch.load(config["output"]["actor_output"].format(seed))))
    model.baseline_network.load_state_dict(torch.load(config["output"]["critic_output"].format(seed)))


    for i_episode in range(1):  # Run for a few episodes
        state, info = env.reset()
        env = gym.wrappers.RecordVideo(
            env,
            config["output"]["record_path"].format(seed),
            step_trigger=lambda x: x % 100 == 0,
        )
        rewards = []
        states = []
        print(state)
        for t in range(1000):  # Maximum number of steps per episode

            action = model.policy.act(state)
            states.append(state)
            state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated:
                print("done, reward:", sum(rewards))
                print(rewards)
                break
    
    print(len(states))
    np.save("state_list",np.stack(states))
    env.close()



            