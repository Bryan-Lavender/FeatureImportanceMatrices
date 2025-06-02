import numpy as np
import torch

import os
from ..utils.general import get_logger, Progbar, export_plot
from ..utils.network_utils import np2torch
from ..ActorCritic.baseline_network import BaselineNetwork
from ..ActorCritic.mlp import build_mlp
from ..ActorCritic.policy import CategoricalPolicy, GaussianPolicy


class PolicyGradient(object):
    """
    Class for implementing a policy gradient algorithm

    Initialize Policy Gradient Class

    Args:
            env_runner (method): method to sample an exectution of the policy
            env_recorder (method): method to record and save an execution of the policy
            config (dict): class with hyperparameters
            logger (): logger instance from the logging module
            seed (int): fixed seed
    """

    def __init__(self, env_runner, env_recorder, config, seed, logger=None):
        # directory for training outputs
        if not os.path.exists(config["output"]["output_path"].format(seed)):
            os.makedirs(config["output"]["output_path"].format(seed))

        # store hyperparameters
        self.config = config
        self.seed = seed

        self.logger = logger
        if logger is None:
            self.logger = get_logger(config["output"]["log_path"].format(seed))
        

        # discrete vs continuous action space
        self.discrete = config["env"]["discrete"]
        self.observation_dim = config["env"]["obs_dim"]
        self.action_dim = config["env"]["action_dim"]

        self.lr = self.config["hyper_params"]["learning_rate"]

        self.device = torch.device("cpu")
        if config["model_training"]["device"] == "cuda" or config["model_training"]["device"] == "gpu":
            if torch.cuda.is_available(): 
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")

        self.init_policy()

        if config["model_training"]["use_baseline"]:
            self.baseline_network = BaselineNetwork(config).to(self.device)

        try:
            if self.config["model_training"]["compile"] == True:
                self.network = torch.compile(self.network, mode=self.config["model_training"]["compile_mode"])
                self.policy = torch.compile(self.policy, mode=self.config["model_training"]["compile_mode"])
                if config["model_training"]["use_baseline"]:
                    self.baseline_network = torch.compile(self.baseline_network, mode=self.config["model_training"]["compile_mode"])
                print("Model compiled")
        except Exception as err:
            print(f"Model compile not supported: {err}")
        self.env_runner = env_runner
        self.env_recorder = env_recorder
        print("device: ",self.device)
    def init_policy(self):
        self.network = build_mlp(self.observation_dim, self.action_dim, self.config['hyper_params']['n_layers'], self.config['hyper_params']['layer_size'])
        self.network.to(self.device)

        if self.discrete:
            self.policy = CategoricalPolicy(self.network, self.device)
        else:
            self.policy = GaussianPolicy(self.network, self.action_dim, self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
       

    def init_averages(self):
        """
        initialize averages
        """
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.
       

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        pass

    def sample_path(self, num_episodes=None):
        """
        Sample paths (trajectories) from the environment. expects a list of dictionaries following 
        path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
            } 
        for a single episode
        Args:
            num_episodes (int): the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            

        Returns:
            paths (list): a list of paths. Each path in paths is a dictionary with
                        path["observation"] a numpy array of ordered observations in the path
                        path["actions"] a numpy array of the corresponding actions in the path
                        path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards (list): the sum of all rewards encountered during this "path"

        
        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        if num_episodes != None:
            while (num_episodes and t < num_episodes):
                path = self.env_runner()
                paths.append(path)
                episode_rewards.append(sum(path["reward"]))
                t += len(path["observation"])
        else:
            while t < self.config["hyper_params"]["batch_size"]:
                path = self.env_runner()
                paths.append(path)
                episode_rewards.append(sum(path["reward"]))
                t += len(path["observation"])

        return paths, episode_rewards

    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        Args:
            paths (list): recorded sample paths. See sample_path() for details.

        Return:
            returns (np.array): return G_t for each timestep

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):

           G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

        where T is the last timestep of the episode.

        Note that here we are creating a list of returns for each path

        """

        all_returns = []
        for path in paths:
            rewards = path["reward"]
      
            returns = []

            # Initialize the return for the last timestep
            G = 0
            # Iterate backwards through the rewards to compute the return
            for r in reversed(rewards):
                G = r + self.config['hyper_params']['gamma'] * G
                returns.insert(0, G)  # Insert at the beginning
           
            all_returns.append(returns)
        returns = np.concatenate(all_returns)

        return returns

    def normalize_advantage(self, advantages):
        """
        Normalized advantages

        Args:
            advantages (np.array): (shape [batch size])
        Returns:
            normalized_advantages (np.array): (shape [batch size])

        

        Note:
        This function is called only if self.config["model_training"]["normalize_advantage"] is True.
        """
       
        mean_advantage = np.mean(advantages)
        std_advantage = np.std(advantages)
        normalized_advantages = (advantages - mean_advantage) / std_advantage

        
        return normalized_advantages

    def calculate_advantage(self, returns, observations):
        """
        Calculates the advantage for each of the observations

        Args:
            returns (np.array): shape [batch size]
            observations (np.array): shape [batch size, dim(observation space)]

        Returns:
            advantages (np.array): shape [batch size]
        """
        if self.config["model_training"]["use_baseline"]:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns

        if self.config["model_training"]["normalize_advantage"]:
            advantages = self.normalize_advantage(advantages)

        return advantages

    def update_policy(self, observations, actions, advantages):
        """
        Args:
            observations (np.array): shape [batch size, dim(observation space)]
            actions (np.array): shape [batch size, dim(action space)] if continuous
                                [batch size] (and integer type) if discrete
            advantages (np.array): shape [batch size]


        TODO:
            Perform one update on the policy using the provided data.
            To compute the loss, you will need the log probabilities of the actions
            given the observations. Note that the policy's action_distribution
            method returns an instance of a subclass of
            torch.distributions.Distribution, and that object can be used to
            compute log probabilities.
            See https://pytorch.org/docs/stable/distributions.html#distribution

        Note:
            PyTorch optimizers will try to minimize the loss you compute, but you
            want to maximize the policy's performance.
        """
        observations = np2torch(observations, device=self.device)
        actions = np2torch(actions, device=self.device)
        advantages = np2torch(advantages, device=self.device)
        ### START CODE HERE ###
        self.optimizer.zero_grad()
        action_dist = self.policy.action_distribution(observations)
        log_probs = log_probs = action_dist.log_prob(actions)
        loss = -(log_probs * advantages).mean()
        loss.backward()
        self.optimizer.step()
        ### END CODE HERE ###

    def train(self):
        """
        Performs training, you do not have to change or use anything here, but it is worth taking
        a look to see how all the code you've written fits together.
        """
        last_record = 0

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        # set policy to device
        self.policy = self.policy.to(self.device)

        for t in range(self.config["hyper_params"]["num_batches"]):

            # collect a minibatch of samples
            paths, total_rewards = self.sample_path()
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)

            # advantage will depend on the baseline implementation
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            if self.config["model_training"]["use_baseline"]:
                self.baseline_network.update_baseline(returns, observations)
            self.update_policy(observations, actions, advantages)

            # logging
            if t % self.config["model_training"]["summary_freq"] == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "{:06d}/{:06d} : Average reward: {:04.2f} +/- {:04.2f}".format(
                t, self.config["hyper_params"]["num_batches"], avg_reward, sigma_reward
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            if self.config["env"]["record"] and (
                last_record > self.config["model_training"]["record_freq"]
            ):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")

        torch.save(
            self.policy.state_dict(),
            self.config["output"]["actor_output"].format(self.seed),
        )
        torch.save(
            self.baseline_network.state_dict(),
            self.config["output"]["critic_output"].format(self.seed),
        )
        np.save(
            self.config["output"]["scores_output"].format(self.seed),
            averaged_total_rewards,
        )
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config["env"]["env_name"],
            self.config["output"]["plot_output"].format(self.seed),
        )

    def evaluate(self, env=None, num_episodes=1, logging=False):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        
        paths, rewards = self.sample_path(num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        if logging:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward
            )
            self.logger.info(msg)
        return avg_reward

    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        self.env_recorder()

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        if self.config["env"]["record"]:
            self.record()
        # model
        self.train()
        # record one game at the end
        if self.config["env"]["record"]:
            self.record()
