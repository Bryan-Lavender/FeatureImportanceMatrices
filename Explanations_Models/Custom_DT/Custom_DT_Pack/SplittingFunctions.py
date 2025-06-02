import numpy as np
#the way this goes is buckets are a dictionary of lists, var is the key and the list is the bucketed value


#discrete variables
def entropy(Y):
    entropy = 0
    if len(Y) == 1:
        return 0
    for i in Y[Y.keys()[0]].unique():
        entropy += -len(Y[Y[Y.keys()[0]] == i])/len(Y[Y.keys()[0]]) * np.log2(len(Y[Y[Y.keys()[0]] == i])/len(Y[Y.keys()[0]]))
    return entropy

def MSE(Y):
    if len(Y) == 1:
        return 0
    Y = Y.to_numpy()
    ym = (1/len(Y)) * Y.sum(axis=0)
    return (1/len(Y)) * np.linalg.norm(Y-ym, axis=1).sum()

def MAE(Y):
    if len(Y) == 1:
        return 0
    Y = Y.to_numpy()
    ym = np.median(Y, axis=0)
    return (1/len(Y)) * np.mean(np.abs(Y-ym), axis=1).sum()

