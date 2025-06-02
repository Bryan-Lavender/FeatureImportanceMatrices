import torch
import numpy as np

class FiniteDifferences():
    def __init__(self, model):
        self.model = model
        

    def Relevence(self, x, h = .5, out_mat = True):
        with torch.no_grad():
            jac = []
            
            for i in range(x.size()[0]):
                x[i] = x[i] + h
                partial = self.model.forward(x)
                x[i] = x[i] - 2*h
                partial = (partial - self.model.forward(x))/(2*h)
                x[i] = x[i] + h
                jac.append(partial.cpu().numpy())
            if out_mat:
                out = self.model.forward(x).cpu().numpy()
            
            jac = np.stack(jac).T
        
            if out_mat:
                return (out, jac)
            return jac
    
    def get_FI(self, X, h=.5, out = False):
        X = torch.tensor(X, dtype = torch.float32)
        jacs = []
        outs = []
        for i in X:
            outt, jac = self.Relevence(i, h=h, out_mat = out)
            jacs.append(jac)
            outs.append(outt)
        return (np.stack(outs), np.stack(jacs))
