import numpy as np
import torch
import torch.nn as nn
from ..utils.network_utils import np2torch
from ..ActorCritic.mlp import build_mlp


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network

    Args:
       
        config (dict): A dictionary containing generated from reading a yaml configuration file

    """

    def __init__(self,  config):
        super().__init__()
        self.config = config
      
        self.lr = self.config["hyper_params"]["learning_rate"]
        self.device = torch.device("cpu")
        if self.config["model_training"]["device"] == "gpu" or self.config["model_training"]["device"] == "cuda":
            if torch.cuda.is_available(): 
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")

   
        input_size = config["env"]["obs_dim"]
        output_size = 1  
        self.network = build_mlp(input_size, output_size, config['hyper_params']['n_layers'], config['hyper_params']['layer_size']).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config['hyper_params']['learning_rate'])
        

    def forward(self, observations):
        """
        Pytorch forward method used to perform a forward pass of inputs(observations)
        through the network

        Args:
            observations (torch.Tensor): observation of state from the environment
                                        (shape [batch size, dim(observation space)])

        Returns:
            output (torch.Tensor): networks predicted baseline value for a given observation
                                (shape [batch size])

    
        """

        output = self.network(observations)
        output = output.squeeze()
  
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """


        Args:
            returns (np.array): the history of discounted future returns for each step (shape [batch size])
            observations (np.array): observations at each step (shape [batch size, dim(observation space)])

        Returns:
            advantages (np.array): returns - baseline values  (shape [batch size])

        """
        observations = np2torch(observations, device=self.device)
        with torch.no_grad():
            baseline_values = self.forward(observations)
        baseline_values = baseline_values.cpu().numpy()
        advantages = returns - baseline_values
        return advantages

    def update_baseline(self, returns, observations):
        """
        Performs back propagation to update the weights of the baseline network according to MSE loss

        Args:
            returns (np.array): the history of discounted future returns for each step (shape [batch size])
            observations (np.array): observations at each step (shape [batch size, dim(observation space)])

     
        """
        returns = np2torch(returns, device=self.device)
        observations = np2torch(observations, device=self.device)

        self.optimizer.zero_grad()
        baseline_predictions = self.network(observations)
        loss_fn = nn.MSELoss()
        loss = loss_fn(baseline_predictions.squeeze(), returns)
        loss.backward()
        self.optimizer.step()