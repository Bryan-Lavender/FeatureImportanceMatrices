from .Custom_DT_Pack.DecisionTree import DecisionTree
import torch
import os
import numpy as np
import json
from .sampling_methods import Policy_Sampler, Uniform_Sampler, Gaussian_Sampler
class SurrogateModel():
    def __init__(self, config, FI_calc = None):
        self.model = DecisionTree(config, FI=FI_calc)
        self.config = config
    
    def fit(self,X,Y,FI_in = None, out_logits=None):
        self.model.fit(X,Y,FI_in = FI_in, out_logits=out_logits)

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
    
    ##TODO
    ## DO NOT TRUST
    def depth_breadth(self):
        tree = self.model
        depth_max = tree.max_depth
        # nodes = [0]
        # widths = [1]
        # for depth in range(depth_max):
        #     new_nodes = []
        #     for node in nodes:
        #         right = tree.root.right_node[node]
        #         if not right == -1:
        #             new_nodes.append(right)
        #         left = tree.root.left_node[node]
        #         if not left == -1:
        #             new_nodes.append(left)
        #     widths.append(len(new_nodes))
        #     nodes = new_nodes
        return (depth_max, 0)
        #return (depth_max, max(widths))
    
    def get_top_split(self):
        return {"feature": self.model.root.feature_index, "range": self.model.root.val_bucket}
    
    def Save(self, FilenameEnder="tree.json"):
        if self.config["surrogate"]["multi_tree"]:
            path = "SavedCustomTreeMets/"+FilenameEnder
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as filename:
                json.dump(self.model.dictionary_reps, filename, indent=4)
        else:
            path = "SavedCustomTreeMets/"+FilenameEnder
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as filename:
                json.dump(self.model.dictionary_rep, filename, indent=4)
    
    def tree_traversal(self):
        print(self.model.get_dict_representation())
    
samplers = {
    "Policy"  : Policy_Sampler,
    "Uniform" : Uniform_Sampler,
    "Gaussian": Gaussian_Sampler,
}

class LIME():
    def __init__(self, config, env_runner, FI_getta = None):
        self.config = config
        self.comparitor = env_runner.comparitor
        self.env_runner = env_runner.runner
        self.surr_model = SurrogateModel(self.config, FI_getta)
        self.FI_getta = FI_getta
        if config["sampler"]["use_dist"]:
            self.deep_model = env_runner.model.policy
        else:
            self.deep_model = env_runner.model.network
        self.device = env_runner.model.device
        self.sampler = samplers[config["sampler"]["sample_type"]](config, self.env_runner)
        
    def sample_set(self):
        with torch.no_grad():
            samples = self.sampler.sample().to(self.device)
            if self.config["surrogate"]["classifier"]:
                Y = torch.argmax(self.deep_model.forward(samples), dim = 1).to("cpu").numpy()
                
            else:
                Y = self.deep_model.forward(samples).to("cpu").numpy()
                
            samples = samples.to("cpu").numpy()
            X = samples
        return X, Y
    def train(self, returner = False):
        with torch.no_grad():
            samples = self.sampler.sample().to(self.device)
            if self.config["surrogate"]["classifier"]:
                Y = torch.argmax(self.deep_model.forward(samples), dim = 1).to("cpu").numpy()
                
            else:
                Y = self.deep_model.forward(samples).to("cpu").numpy()
                
            samples = samples.to("cpu").numpy()
            X = samples
            if self.config["surrogate"]["use_FI"] and self.FI_getta != None:
                self.surr_model.fit(X,Y)
            else:
                self.surr_model.fit(X,Y)
        if returner:
            return X,Y
    #can only test action and state metrics
    def percent_Correct(self, print_val = False):
        PCs = []
        for i in range(0, 30):
            path = self.env_runner(use_dist = self.config["sampler"]["use_dist"])
            y_pred = self.surr_model.forward(path["observation"])
        
            y_true = path["action"]
            
            TP = 0
            true_positive = sum(yp == yt for yp, yt in zip(y_pred, y_true))
            percent_correct = true_positive/len(y_true)
            if self.config["surrogate"]["classifier"]:
            #     for i in list(range(self.config["env"]["action_dim"])):
            #         print(i)
            #         print((y_true == i) & (y_pred == i))
                    
            #         TP += np.sum((y_true == i) & (y_pred == i))
            #     total = len(y_true)
            #     percent_correct = TP/total
                
                if(print_val):
                    print(f"Percent Correct: {percent_correct:.2f}%")
                PCs.append(percent_correct)
            else:
                val = np.mean((y_pred - y_true)**2)
                PCs.append(val)
        return np.mean(PCs)

    
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
        if self.config["env"]["discrete"]:
            output = np.argmax(output, axis = 1)
            matches = surr_output == output
            num_matches = np.sum(matches)
            percentage_correct = (num_matches / len(surr_output)) 
            return percentage_correct
        else:
            return np.mean(((surr_output - output)**2).mean(axis=0))

        
    
    def absolute_distance(self, print_val= True):
        return self.comparitor(use_dist=self.config["sampler"]["use_dist"], model2 = self.surr_model)
    
    def get_metrics(self):
        mets = self.surr_model.model.get_metrics()
        global_mets = {"Policy_Captured": float(self.percent_Correct()), "Uniform_Captured": float(self.uniform_Correct())}
        global_mets.update(mets)
        return global_mets


    