import pandas as pd
import numpy as np
from .Node import Single_Attribute_Node
class DecisionTree():
    def __init__(self, config, FI=None):
        self.config = config
        self.FI= FI
    
        self.single_tree = not config["surrogate"]["multi_tree"]
        self.FI_set = False
    def fit(self, X,Y, FI_in = None, out_logits = None):
   
        if self.single_tree:
            self.fit_single(X,Y, FI_in = FI_in, out_logits = out_logits)
        else:
          
            self.fit_multi_tree(X,Y, FI_in = FI_in, out_logits = out_logits)

    #MAKING IT NOT TURN INTO DF TEMPORARILY FOR METRICS
    def fit_multi_tree(self, X,Y, FI_in = None, out_logits = None):
        X = pd.DataFrame(X, columns=self.config["picture"]["labels"])
        Y_set =[]
        if self.config["env"]["discrete"]:
            self.config["surrogate"]["classifier"] = False
            self.config["surrogate"]["criterion"] = "MSE"
            Y = out_logits
            for i in self.config["picture"]["class_names"]:
                Y_ind = Y[[i]]
                
                Y_set.append(Y_ind)
        else:
            # for i in range(Y.shape[1]):
            #     Y_ind = Y[:,i]
            for i in self.config["picture"]["class_names"]:
                Y_ind = Y[[i]]
                #Y_ind = pd.DataFrame(Y_ind, columns=[self.config["picture"]["class_names"][i]])
                
                Y_set.append(Y_ind)
        if self.config["surrogate"]["use_FI"]:
            if type(FI_in) == type(None):
                out, FI = self.get_FI(X)
            else:
                FI=FI_in
                out = out_logits
            #self.dictionary_rep, self.max_depth = self.root.fit(X,Y,0, FI = FI, out_logits = out)
            self.roots = {}
            self.max_depths = {}
            self.dictionary_reps = {}
            for i in Y_set:
                self.roots[i.columns[0]] = Single_Attribute_Node(self.config)
                tmp_dict, tmp_max = self.roots[i.columns[0]].fit(X,i,0,FI=FI[i.columns[0]], out_logits=out[i.columns[0]])
                self.dictionary_reps[i.columns[0]] = tmp_dict 
                self.max_depths[i.columns[0]] = tmp_max
            
        else:    
            #self.dictionary_rep, self.max_depth = self.root.fit(X,Y,0)
         
            self.roots = {}
            self.max_depths = {}
            self.dictionary_reps = {}
            for i in Y_set:
                
                self.roots[i.columns[0]] = Single_Attribute_Node(self.config)
                tmp_dict, tmp_max = self.roots[i.columns[0]].fit(X,i,0)
                self.dictionary_reps[i.columns[0]] = tmp_dict 
                self.max_depths[i.columns[0]] = tmp_max
        self.nodeList = self.node_list()
        if self.config["env"]["discrete"]:
            self.config["surrogate"]["classifier"] = True
            self.config["surrogate"]["criterion"] = "entropy"
    def fit_single(self, X,Y, FI_in=None, out_logits = None):
        X = pd.DataFrame(X, columns=self.config["picture"]["labels"])
        if self.config["surrogate"]["classifier"]:
            Y = pd.DataFrame(Y, columns=["out"])
        else:
            Y = pd.DataFrame(Y, columns=self.config["picture"]["class_names"])
        self.root = Single_Attribute_Node(self.config)
        if self.config["surrogate"]["use_FI"]:
            if type(FI_in) == type(None):
                out, FI = self.get_FI(X)
            else:
                out = out_logits
                FI = FI_in
            
            self.dictionary_rep, self.max_depth = self.root.fit(X,Y,0, FI = FI, out_logits = out)
        

        else:    
            self.dictionary_rep, self.max_depth = self.root.fit(X,Y,0)

        self.nodeList = self.node_list()
    def get_FI(self,X):
        out, FI = self.FI.Relevence(X)
        FI = np.abs(FI)
        columns = pd.MultiIndex.from_product([self.config["picture"]["class_names"], self.config["picture"]["labels"]], names=['OutLogit', 'InLogit'])

        # Reshape the data to (60, 8) to match the multi-level column structure
        reshaped_data = FI.reshape(FI.shape[0], -1)

        # Create the DataFrame
        FI = pd.DataFrame(reshaped_data, columns=columns)
        out = pd.DataFrame(out, columns=self.config["picture"]["class_names"])
        return out,FI
        
    def _forward(self, val):
        if self.single_tree:
            return self.root._forward(val)
        else:
            re = []
            for i in self.config["picture"]["class_names"]:
                re.append(self.roots[i]._forward(val)[0])
            re = np.array(re)
            if self.config["env"]["discrete"]:
                
                re = np.argmax(re)
                
            return re
    
    def predict(self, vals):
        X = pd.DataFrame(vals, columns=self.config["picture"]["labels"])

        results = []
        for index, row in X.iterrows():
            results.append(self._forward(row))

        return results
    def get_dict_representation(self):
        return self.dictionary_rep
    def printer(self):
        print("root")
        print(self.root.printer())
    
    def TraverseTree(self,node:Single_Attribute_Node):
        if node.is_leaf:
            return [node]
        else:
            re = []
            left = self.TraverseTree(node.left_node)
            right = self.TraverseTree(node.right_node)
            for i in left:
                re.append(i)
            for i in right:
                re.append(i)
            return re
    
    def node_list(self):
        if self.config["surrogate"]["multi_tree"]:
            nodes = {}
            for i in self.config["picture"]["class_names"]:
                nodes[i] = self.TraverseTree(self.roots[i])
            return nodes
        return self.TraverseTree(self.root)
    
    def get_node_list(self):
        return self.nodeList
    
    def get_depth(self):
        if self.config["surrogate"]["multi_tree"]:
            return self.max_depths
        return self.max_depth
    
    def get_breadth(self):
        if self.config["surrogate"]["multi_tree"]:
            breadths = {}
            for i in self.config["picture"]["class_names"]:
                
                breadths[i] = self.get_breadth_multi(i)
            return breadths
        else:
            return self.get_breadth_single()
    def get_breadth_single(self):
        breadths = [0 for i in range(self.max_depth + 1)]
        for node in self.nodeList:
            breadths[node.depth] += 1
        return max(breadths)
    
    def get_breadth_multi(self,clas):
   
        breadths = [0 for i in range(self.max_depths[clas] + 1)]
        for node in self.nodeList[clas]:
            breadths[node.depth] += 1
        return max(breadths)
    
    def get_avg_representation(self):
        if self.config["surrogate"]["multi_tree"]:
            reps = {}
            for i in self.config["picture"]["class_names"]:
                reps[i] = self.get_avg_represenetation_multi(i)
            return reps
        else:
            return self.get_avg_representation_single()
    def get_avg_represenetation_multi(self, clas):
        Running_Avg = 0
        for node in self.nodeList[clas]:

            Running_Avg += node.represented_nodes
        return Running_Avg/len(self.nodeList[clas])
    def get_avg_representation_single(self):
        Running_Avg = 0
        
        for node in self.nodeList:
            Running_Avg += node.represented_nodes
        return Running_Avg/len(self.nodeList)
    
    def get_metrics(self):
        return {"Representation": self.get_avg_representation(), "Depth": self.get_depth(), "Breadth": self.get_breadth()}
    
    
        
