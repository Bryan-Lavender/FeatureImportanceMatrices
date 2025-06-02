import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from array import array
import random
class Gaussian_Sampler():
    def __init__(self, config, runner):
        self.config = config

    def sample(self, num = None, mean = None):
        
        if num == None:
            num_samples = self.config["sampler"]["num_samples"]
        else:
            num_samples = num
        if type(mean) == int:
            return torch.normal(mean = mean, std = self.config["sampler"]["std"], size = ( num_samples,self.config["env"]["obs_dim"]))
        else:
            means_repeated = mean.repeat(1, num_samples, 1)
            samples = torch.normal(means_repeated, self.config["sampler"]["std"])
            return samples

class Uniform_Sampler():
    def __init__(self, config, runner):
        self.config = config
    def sample(self, num = None):
        if num == None:
            num_samples = self.config["sampler"]["num_samples"]
        else:
            num_samples = num
        bounds = np.transpose(self.config["sampler"]["bounds"])
        bounds = torch.tensor(bounds, dtype = torch.float32)
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        uniform_samples = torch.rand((num_samples, bounds.shape[0]))
        scaled_samples = lower_bounds + uniform_samples * (upper_bounds - lower_bounds)
        return scaled_samples
    
class Policy_Sampler():
    def __init__(self, config, runner):
        self.config = config
        self.runner = runner

    def sample(self, num = None):
        if num == None:
            num_samples = self.config["sampler"]["num_samples"]
        else:
            num_samples = num
        samples = []
        while num_samples > len(samples):
            path = self.runner(use_dist = self.config["sampler"]["use_dist"])
            samples.extend(path["observation"])

        samples_indicies = random.sample(range(len(samples)), num_samples)
        return torch.tensor(samples, dtype = torch.float32)[samples_indicies]


