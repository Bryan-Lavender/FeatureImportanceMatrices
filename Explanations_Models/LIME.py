import torch
from Explanations_Models.sampling_methods import GaussianSampler, UniformSampler, UniformPolicySampler
from Explanations_Models.surrogate_models import LassoRegression, DecisionTreeSur
import numpy as np


samplers = {
    'GaussianSampler': GaussianSampler,
    'UniformSampler' : UniformSampler,
    'UniformPolicySampler': UniformPolicySampler
}

models = {
    'LassoRegression': LassoRegression,
    'DecisionTree'   : DecisionTreeSur
}


class LimeModel():
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
        
        
        
        if config['surrigate_params']['model_alg'] == "REG":
            self.interpretable_models = []
            for i in range(0, config['sampling']['output_num']):
                self.interpretable_models.append(self.surrogate_mod(self.point,config['surrigate_params']).to(config["model_training"]["device"]))
        else:
            self.interpretable_models = self.surrogate_mod(self.point,config['surrigate_params'])

        
        self.sample_points = None
    
    def sample(self, config):
        self.sample_points = torch.tensor(self.sampler.sample(config), device=self.config["model_training"]["device"], dtype=torch.float32)

    
    def fit_surrigate(self, X, Y):
        out = 0
        for i in self.interpretable_models:
            i.fit(X, Y[:, out])
            out += 1
    
    def runner(self):
        self.sample(self.config['sampling'])
        with torch.no_grad():
            Y = self.model(self.sample_points)
        self.fit_surrigate(self.sample_points, Y)
        
        ct = 0
        acc_arr = []
        for i in self.interpretable_models:
            torch.save(i.linear.state_dict(), self.config["explanation_weights"]["model_path"] + self.config["explanation_weights"]["outputs"][ct] + ".pt")
            acc_arr.append(i.evaluate(self.sample_points, Y[:, ct]))
            ct += 1
        np.save(self.config["explanation_output"]["MAE_MSE_RMSE_Rsq"],np.array(acc_arr))

    
        
        
