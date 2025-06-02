from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .sampling_methods import Policy_Sampler, Uniform_Sampler, Gaussian_Sampler
import torch
import numpy as np
from sklearn.tree import plot_tree
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
class DecisionTree():
    def __init__(self, config, env_runner):
        if config["surrogate"]["classifier"]:
            self.model = DecisionTreeClassifier(criterion=config["surrogate"]["criterion"], min_samples_split=config["surrogate"]["min_split"])
        else:
            self.model = DecisionTreeRegressor(criterion=config["surrogate"]["criterion"], min_samples_split=config["surrogate"]["min_split"])
        
        self.env_runner = env_runner
        self.config = config
    
    def fit(self,X, Y):
        self.model.fit(X, Y)
    
    def forward(self, X):
        is_pass = False
        if type(X) == torch.Tensor:
            X = X.to("cpu").numpy()
        if len(X.shape) == 1:
            is_pass = True
            X = np.array([X])
            if self.config["surrogate"]["classifier"] and self.config["env"]["discrete"]:
                return self.model.predict(X)[0]
            elif not self.config["surrogate"]["classifier"] and self.config["env"]["discrete"]:
                np.argmax(self.model.predict(X)[0])
            else:
                return self.model.predict(X)[0]
            
        return self.model.predict(X)

    def display_tree(self):
        fig = plt.figure(figsize=((25,20)))
        plot_tree(self.model, 
                feature_names = self.config["picture"]["labels"],
                class_names = self.config["picture"]["class_names"],
                impurity=False,
                proportion=True,
                filled=True
                )
        fig.savefig("daTreeMan.png")
    
    def depth_breadth(self):
        tree = self.model
        depth_max = tree.tree_.max_depth
        nodes = [0]
        widths = [1]
        for depth in range(depth_max):
            new_nodes = []
            for node in nodes:
                right = tree.tree_.children_right[node]
                if not right == -1:
                    new_nodes.append(right)
                left = tree.tree_.children_left[node]
                if not left == -1:
                    new_nodes.append(left)
            widths.append(len(new_nodes))
            nodes = new_nodes
        
        return (depth_max, max(widths))
    
    def get_top_split(self):
        return self.model.tree_.feature[0]
    
    def Save(self, FilenameEnder = "tree.pkl"):
        path = "SavedTrees/"+self.config["env"]["env_name"]+"/"+self.config["sampler"]["sample_type"]+"/"+self.config["surrogate"]["criterion"]+"/"+str(self.config["sampler"]["num_samples"])+"/"+FilenameEnder
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as filename:
            pickle.dump(self.model, filename)
        return path
    
    def Load(self, filename, path = None):
        if path == None:
            filename = "SavedTrees/"+self.config["sampler"]["sample_type"]+"/"+self.config["surrogate"]["criterion"]+"/"+str(self.config["sampler"]["num_samples"])+"/"+filename
        else:
            filename= path + "/" + filename
        
        with open(filename, "rb") as model_file:
            self.model = pickle.load(model_file)
    
    



samplers = {
    "Policy"  : Policy_Sampler,
    "Uniform" : Uniform_Sampler,
    "Gaussian": Gaussian_Sampler,
}

class LIME():
    def __init__(self, config, env_runner):
        self.config = config
        self.comparitor = env_runner.comparitor
        self.env_runner = env_runner.runner
        self.surr_model = DecisionTree(self.config, self.env_runner)
        if config["sampler"]["use_dist"]:
            self.deep_model = env_runner.model.policy
        else:
            self.deep_model = env_runner.model.network
        self.device = env_runner.model.device
        self.sampler = samplers[config["sampler"]["sample_type"]](config, self.env_runner)
        
    
    def train(self):
        with torch.no_grad():
            samples = self.sampler.sample().to(self.device)
            if self.config["surrogate"]["classifier"]:
                Y = torch.argmax(self.deep_model.forward(samples), dim = 1).to("cpu").numpy()
                arr = [len(Y[Y==i]) for i in range(self.config["env"]["action_dim"])]
            else:
                Y = self.deep_model.forward(samples).to("cpu").numpy()
                arr = []
            samples = samples.to("cpu").numpy()
            X = samples
            self.surr_model.fit(X,Y)
    
    #can only test action and state metrics
    def percent_Correct(self, print_val = False):
        path = self.env_runner(use_dist = self.config["sampler"]["use_dist"])
        y_pred = self.surr_model.forward(path["observation"])
        y_true = path["action"]
        TP = 0
        if self.config["surrogate"]["classifier"]:
            for i in range(self.config["env"]["action_dim"]):
                TP += np.sum((y_true == i) & (y_pred == i))
            total = len(y_true)
            percent_correct = TP/total
            if(print_val):
                print(f"Percent Correct: {percent_correct:.2f}%")
            return percent_correct
        else:
            val = np.mean((y_pred - y_true)**2)
            return val

    
    def uniform_Correct(self, print_val = False, num = 10000):
        sample_path = "uniform_samples/"+self.config["env"]["env_name"]+"/input_samples.npy"
        out_path = "uniform_samples/"+self.config["env"]["env_name"]+"/output_samples.npy"
        if os.path.exists(sample_path):
            samples = np.load(sample_path)
            output = np.load(out_path)
        else:
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            sampler = Uniform_Sampler(self.config, runner = None)
            samples = sampler.sample(num = num).to(self.device)
            
            with torch.no_grad():
                output = self.deep_model.forward(samples).to("cpu").numpy()
                np.save(out_path, output)
            np.save(sample_path, samples.to("cpu").numpy())
        surr_output = self.surr_model.forward(samples)
        if self.config["surrogate"]["classifier"]:
            output = np.argmax(output, axis = 1)
            matches = surr_output == output
            num_matches = np.sum(matches)
            percentage_correct = (num_matches / len(surr_output)) 
            return percentage_correct
        else:
            return np.mean(((surr_output - output)**2).mean(axis=0))

        
    
    def absolute_distance(self, print_val= True):
        return self.comparitor(use_dist=self.config["sampler"]["use_dist"], model2 = self.surr_model)
    
    

        
    
        

    
    
        

    