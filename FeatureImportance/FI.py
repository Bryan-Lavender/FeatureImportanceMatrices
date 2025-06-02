import torch
from torch import nn
import numpy as np
from .methods.LRP import TwoDWeights_LRPModel
from .methods.FiniteDifferences import FiniteDifferences
methods = {
    "LRP": TwoDWeights_LRPModel,
    "FD": FiniteDifferences
}
class FeatureImportance():
    def __init__(self, config, model):
        self.config = config
        self.model = model
        if self.config["model_training"]["device"] == "gpu":
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        
    
    def Calc_Relevence_Importance(self, input_point, h = .5, printer = False):
        if self.config["FI"]["method"] == "LRP":
            Model = methods[self.config["FI"]["method"]](self.model)
            out, FI_matrix = Model.get_FI(input_point, printer)
            FI_matrix = torch.stack(FI_matrix).detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            return out, FI_matrix
        else:
            Model = methods[self.config["FI"]["method"]](self.model)
            out, FI_matrix = Model.Relevence(input_point, h = h)
           
            return out, FI_matrix
        
    def Relevence(self, X, h=10e-5):
        #print(h)
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        FI = []
        out = []
        for i in X:
            out_t, tmpFI = self.Calc_Relevence_Importance(i, h=h)
            out.append(out_t)
            tmpFI = np.abs(tmpFI)
            
            # for j in range(tmpFI.shape[0]):
            #     tmpFI[j,:]= (tmpFI[j,:] - np.min(tmpFI[j,:])) / (np.max(tmpFI[j,:]) - np.min(tmpFI[j,:]))
            FI.append(tmpFI)
        if self.config["normalize_FI"]:
            FI = np.concatenate(FI)
            FI = (FI - np.min(FI)) / (np.max(FI) - np.min(FI))
            FI = np.split(FI, len(X))
    
        FI = np.stack(FI)
        out = np.stack(out)
        return out, FI
    


        

