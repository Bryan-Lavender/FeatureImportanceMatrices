import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from abc import ABC, abstractmethod
from array import array

class BaseSampler(ABC):
    def __init__(self):
        ABC.__init__(self)
        
    
    @abstractmethod
    def sample(self, N,point, variable_of_interest = "all"):
        """
        this takes in a point (typically in terms of state variables) and samples around it
        point: must be a numpy vector/ vector of vectors
        variable_of_interest: change to get information over the vector being samples
        """
        pass
    def get_samples(self, N, point, variable_of_interest = "all"):
        samples = self.sample(N, point, variable_of_interest)
        
        return samples
    

class GaussianSampler(BaseSampler):
    def __init__(self, variance = 1):
        BaseSampler.__init__(self)
   
        self.variance = variance
    
    def sample(self, N, point, variable_of_interest):
        if variable_of_interest == "all":
            return np.random.normal(loc = point, scale = self.variance,  size=(N, point.shape[0]))
        else:
            sampled_points = np.zeros((N, point.shape[0]))
            for i in range(point.shape[0]):
                if i in variable_of_interest:
                    sampled_points[:, i] = np.random.normal(loc=point[i], scale=self.variance, size=N)
                else:
                    sampled_points[:, i] = point[i]
            return sampled_points

class UniformSampler(BaseSampler):
    def __init__(self):
        BaseSampler.__init__(self)
    def sample(self,config):
        if config['variables'] == "all":
            # Define the bounds as a list of [min, max] pairs for each entry
            bounds = config["bounds"]

            # Initialize an empty list to store the sampled vectors
            sampled_vectors = []

            # Sample uniformly between the bounds for each entry, N times
            for _ in range(config["sample_number"]):
                sampled_vector = np.array([np.random.uniform(low, high) for low, high in bounds])
                sampled_vectors.append(sampled_vector)

            # Convert the list of vectors into a NumPy array
            sampled_vectors = np.stack(sampled_vectors)
            print(sampled_vectors.shape)
            return sampled_vectors
        ##NOT IMPLENETED WITH BOUNDS
        else:
            variable_of_interest = config["variables"]
            point = config["point"]
            N = config["sample_number"]
            sampled_points = np.zeros((N, point.shape[0]))
            for i in range(point.shape[0]):
                if i in variable_of_interest:
                    sampled_points[:, i] = np.random.uniform(loc=point[i], size=N)
                else:
                    sampled_points[:, i] = point[i]
            
            return sampled_points

class UniformPolicySampler(BaseSampler):
    def __init__(self):
        BaseSampler.__init__(self)
    def sample(self, config):
        states = np.load(config["sim_file"])
        samp_ind = np.random.choice(states.shape[0], size = config["sample_number"], replace = False)
        sampled_vects = states[samp_ind]
        print(sampled_vects.shape)
        return sampled_vects