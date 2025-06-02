import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from sklearn import tree

class LassoRegression(nn.Module):
    def __init__(self, point_of_interest, config):
        super().__init__()
        self.point_of_interest = point_of_interest
        self.learning_rate = config['learning_rate']
        self.regularizer = config['regularizer']
        self.num_epochs = config['num_epochs']
        self.linear = nn.Linear(config['input_size'], 1, bias=False)
        self.sigma = config['sigma']
        self.use_dist = config['use_dist']

    
    def forward(self, X):
        out = self.linear(X)
        return out
    
    
    def fit(self, X, Y):
        optimizer = torch.optim.SGD(self.linear.parameters(), lr=self.learning_rate)
        criterion = self.CustLoss
        self.linear.to("cuda")
        for i in range(self.num_epochs):
            y_pred = self.linear(X).squeeze()
            loss = criterion(self.point_of_interest, X, y_pred, Y)
            l1_penalty = self.regularizer * sum(p.abs().sum() for p in self.linear.parameters())
            total_loss = loss + l1_penalty

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        self.MAE, self.MSE, self.RMSE, self.R_Squared = self.evaluate(X,Y)
    
    
    def evaluate(self, X, Y):
        with torch.no_grad():
            MAE = (1/len(X)) * torch.sum(torch.abs(self.forward(X).flatten() - Y))
            MSE = (1/len(X)) * torch.sum((self.forward(X).flatten() - Y)**2)
            RMSE = torch.sqrt((1/len(X)) * torch.sum((self.forward(X).flatten() - Y)**2))
            R_Squared = 1 - (MSE/torch.var(Y))
            return([MAE.item(), MSE.item(), RMSE.item(), R_Squared.item()])

    
    def pis_func(self,point_of_interest, x):
        D = (torch.cdist(point_of_interest, x)**2)/(self.sigma**2)
        return torch.exp(-D)
        
    
    def CustLoss(self, point_of_interest, x, y_pred, Y):
        point_of_interest = torch.tensor(point_of_interest)
        pis = self.pis_func(point_of_interest.unsqueeze(0).to("cuda"), x)
        if self.use_dist:
            return torch.sum(pis*((y_pred - Y)**2))
        else:
            loss = torch.nn.MSELoss()
            return loss(y_pred, Y)


class DecisionTreeSur():
    def __init__(self, point_of_interest, config):
        self.point_of_interest = point_of_interest
        self.learning_rate = config['learning_rate']
        self.regularizer = config['regularizer']
        self.num_epochs = config['num_epochs']
        self.tree = tree.DecisionTreeClassifier()
        self.sigma = config['sigma']
        self.use_dist = config['use_dist']

    
    def fit(self, X, Y):
        X = X.to("cpu").numpy()
        Y = Y.to("cpu").numpy()
        self.tree = self.tree.fit(X,Y)
    
    
    def forward(self, X):
        X = X.to("cpu").numpy()
        return self.tree.predict(X)

    def evaluate(self, X, Y):
        Y = Y.to("cpu").numpy()
        X = self.forward(X)
        ct = 0
        for x,y in zip(X,Y):
            if x == y:
                ct += 1

        ct = ct/Y.shape[0]


        return ([ct,ct, ct, ct])

