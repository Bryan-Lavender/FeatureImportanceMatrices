import torch
from Explanations_Models.sampling_methods import GaussianSampler, UniformSampler, UniformPolicySampler
from Explanations_Models.surrogate_models import LassoRegression, DecisionTreeSur
import numpy as np
import pickle

samplers = {
    'GaussianSampler': GaussianSampler,
    'UniformSampler' : UniformSampler,
    'UniformPolicySampler': UniformPolicySampler
}

models = {
    'LassoRegression': LassoRegression,
    'DecisionTree'   : DecisionTreeSur
}


class LIME():
    def __init__(self, model, point, config):
        self.config = config
        self.model = model.to(config["model_training"]["device"])
        self.point = point
        self.sampler = GaussianSampler(1)

        try:
            self.sampler = samplers[config['sampling']['sampler']]
        except:
            print("Bad Sampler")
            exit(1)
    
        try:
            self.surrogate_mod = models[config['surrigate_params']['model_type']]
        except:
            print("Bad Model")
            exit(1)

        self.interpretable_model = self.surrogate_mod(self.point,config['surrigate_params'])
        self.sample_points = None
    


    def sample(self, config):
        self.sample_points = torch.tensor(self.sampler.sample(config), device=self.config["model_training"]["device"], dtype=torch.float32)


    def fit_surrigate(self, X, Y):
        self.interpretable_model.fit(X, Y)


    def runner(self):
        self.sample(self.config['sampling'])
        with torch.no_grad():
            Y = self.model(self.sample_points)
        self.fit_surrigate(self.sample_points, Y)
        
        ct = 0
        acc_arr = []

        if self.config['surrigate_params']['model_alg'] != "REG":
            pickle.dumps(self.surrogate_mod.tree, self.config["explanation_weights"]["model_path"] + self.config["explanation_weights"]["outputs"][ct] + ".pickle")


    
