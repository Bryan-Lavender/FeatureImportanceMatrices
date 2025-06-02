import pandas as pd
import numpy as np
import torch
"""
FI = Feature Importance
Begin Global Policy Smoothing FI which I am calling the Professa's Method
input: FI pandas frame
output: single FI vector frame where each ~theta~ has a single FI weighting
    Think: x = [x1, x2, ..., xN]
    output: w(x) = [FI_x1, FI_x2, ..., FI_xN]
"""

def Policy_Smoothing_Var_Weighting(FI, out_logits):
    smoothed_fi = FI.mean()
    output_variance = out_logits.var()
    tmp = []
    for i in list(out_logits.keys()):
        if len(tmp) == 0:
            tmp = FI[i].mean().to_numpy() * out_logits[i].var()
        else:
            tmp += FI[i].mean().to_numpy() * out_logits[i].var()
    tmp = pd.DataFrame(np.expand_dims(tmp, axis=0), columns = list(FI[out_logits.keys()[0]].keys()))
    return tmp

def Policy_Smoothing_Max_Weighting(FI, out_logits):

    tmp = []
    for i in list(out_logits.keys()):
        tmp.append(FI[i].mean().to_numpy())
    tmp = np.stack(tmp).max(axis=0)
    tmp = pd.DataFrame(np.expand_dims(tmp, axis=0), columns = list(FI[out_logits.keys()[0]].keys()))
    return tmp

def Classification_Forgiveness(FI, out_logits):

    rows = FI.shape[0] 
    high_labels = FI.columns.get_level_values(0).unique()  
    sub_labels = FI.columns.get_level_values(1).unique() 

    
    FI_np = FI.to_numpy().reshape(rows, len(high_labels), len(sub_labels))
    tmp = FI_np[np.arange(FI_np.shape[0]),np.argmax(out_logits.to_numpy(), axis=1)].mean(axis=0)

    return pd.DataFrame(np.expand_dims(tmp, axis=0), columns = list(FI[out_logits.keys()[0]].keys()))

def Policy_Smoothing_MaxAvg_Weighting(FI, out_logits):

    tmp = []
    for i in list(out_logits.keys()):
        tmp.append(FI[i].mean().to_numpy())
    tmp = tmp[np.argmax(np.stack(tmp).mean(axis=1))]
    tmp = pd.DataFrame(np.expand_dims(tmp, axis=0), columns = list(FI[out_logits.keys()[0]].keys()))
    return tmp
# def PCA_Ish(FI, out_logits):
#     rows = FI.shape[0] 
#     high_labels = FI.columns.get_level_values(0).unique()  
#     sub_labels = FI.columns.get_level_values(1).unique() 

    
#     FI_np = FI.to_numpy().reshape(rows, len(high_labels), len(sub_labels))
#     tmp = FI_np[np.arange(FI_np.shape[0]),np.argmax(out_logits.to_numpy(), axis=1)].mean(axis=0)
#     A = torch.tensor(tmp)
#     vals, vecs = torch.linalg.eig(A.T@A)
#     weighted_FI = torch.unsqueeze(vals.real,1) * vecs.real
#     return torch.sum(weighted_FI, dim=0).detach().numpy()

