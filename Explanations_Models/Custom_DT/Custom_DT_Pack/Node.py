import numpy as np
import pandas as pd
from .BucketAlgs import half_split
from .SplittingFunctions import entropy, MSE, MAE
from scipy.spatial.distance import pdist
from .FI_Calulator import Policy_Smoothing_Max_Weighting, Policy_Smoothing_MaxAvg_Weighting, Policy_Smoothing_Var_Weighting, Classification_Forgiveness
bucketing_mech = {"half_split": half_split}
splitting_functions = {"entropy": entropy, "MSE": MSE, "MAE": MAE}
FICalcs = {"Var_weighted": Policy_Smoothing_Var_Weighting, "Max_all": Policy_Smoothing_Max_Weighting, "Max_avg":Policy_Smoothing_MaxAvg_Weighting, "class": Classification_Forgiveness}

ALPHA = .2
BETA = .8
class Single_Attribute_Node():
    #begin Single_Attribute_Node: X (input), Y (output), feature_list ([disc, cont, disc, disc, cont...])
    """
    Global_Vars: 
        feature_index: index of split
        is_leaf: leaf node to return
        vals: values of bound on feature OR values to return
    """
    def __init__(self, config, parent_node = None):
        self.feature_index = None
        self.is_leaf = False
        self.val_bucket = None
        self.return_value = None
        self.config = config
        self.parent_node = parent_node
        self.represented_nodes = None
        self.FI_calculator = FICalcs
        #if self.config["surrogate"]["multi_tree"]:
            #self.FI_calculator = Single_Y_FICalcs
        
    def should_create_leaf(self, y_values, threshold_factor=2):
        # Calculate the mean of the values
        mean_value = np.mean(y_values)
        
        # Calculate the Euclidean distances from the mean
        distances = np.linalg.norm(y_values - mean_value, axis=1)
        
        # Calculate the standard deviation of the distances
        std_dev = np.std(distances)
        
        # Set the threshold based on standard deviation and vector size
        threshold = threshold_factor * std_dev * np.sqrt(len(y_values[0]))
        
        # Check if all distances are below the threshold
        if np.all(distances < threshold):
            return True  # Create a leaf node
        else:
            return False  # Continue splitting
        
    def fit(self, X,Y, depth=None, FI = None, out_logits = None):
        
        if depth != None:
            self.depth = depth + 1
        else:
            self.depth = 1
        
        if self.config["surrogate"]["classifier"] and len(Y[Y.keys()[0]].unique()) == 1:
            self.represented_nodes = len(Y)
            self.return_value = Y[Y.keys()[0]].value_counts().index[0]
            self.is_leaf = True

            self.left_node = -1
            self.right_node = -1
            return {"Value": int(self.return_value)},self.depth
        
        #elif not self.config["surrogate"]["classifier"] and (len(Y)==1 or max(pdist(Y.values, metric="euclidean")) <= self.config["surrogate"]["tree_cutoff"] ):
        #elif not self.config["surrogate"]["classifier"] and len(Y)==1:# or max(pdist(Y.values, metric="euclidean")) <= self.config["surrogate"]["tree_cutoff"] ):
        #elif not self.config["surrogate"]["classifier"] and (len(Y)==1 or self.depth >= 5):
        #elif not self.config["surrogate"]["classifier"] and (len(Y)==1 or max(pdist(Y.values, metric="euclidean")) <= 2.):
        elif not self.config["surrogate"]["classifier"] and (len(Y) == 1 or self.should_create_leaf(Y.to_numpy())):
            if len(Y) == 1:
                self.return_value = Y.to_numpy()[0]
                self.represented_nodes = len(Y)
            else:
                self.represented_nodes = len(Y)
                self.return_value = np.mean(Y.to_numpy(), axis=0)
                

            self.is_leaf = True

            self.left_node = -1
            self.right_node = -1
            to_return = []
           
            for i in self.return_value:
                to_return.append(float(i))
            return {"Value": to_return},self.depth
        else:
            buckets = self.bucket(X)
            
            min_var, min_bucket, min_val,indicies_left, indicies_right = self.split(X,Y, buckets, FI=FI, out_logits=out_logits)
            self.feature_index = min_var
            self.val_bucket = min_bucket
            self.heuristic_value = min_val
            
     
            self.left_node = Single_Attribute_Node( self.config, parent_node=self)
            self.right_node = Single_Attribute_Node( self.config, parent_node=self)
           
            if self.config["surrogate"]["use_FI"] and type(FI) != type(None):
                left_dict, left_max = self.left_node.fit(X.loc[indicies_left], Y.loc[indicies_left], self.depth, FI=FI.loc[indicies_left], out_logits=out_logits.loc[indicies_left])
                right_dict, right_max = self.right_node.fit(X.loc[indicies_right], Y.loc[indicies_right],self.depth, FI=FI.loc[indicies_right], out_logits=out_logits.loc[indicies_right])
            else:
                left_dict, left_max = self.left_node.fit(X.loc[indicies_left], Y.loc[indicies_left], self.depth)
                right_dict, right_max = self.right_node.fit(X.loc[indicies_right], Y.loc[indicies_right],self.depth)
            
            return {"Feature": self.feature_index, "Bucket": float(self.val_bucket), "Left_Child": left_dict, "Right_Child": right_dict}, max(left_max, right_max)
    
    def bucket(self, X, feature_types = None):
        buckets = {}
        if feature_types == None or self.config == None:
            for key in X.keys():
                if X[key].dtype == "float64" or X[key].dtype == "float32":
                    buckets[key] = bucketing_mech[self.config["surrogate"]["bucket_alg"]](X[key].unique())
                else:
                    buckets[key] = list(X[key].unique())
        else:
            for key in X.keys():
                if self.config["feature_types"][key] == "cont":
                    buckets[key] = bucketing_mech[self.config["surrogate"]["bucket_alg"]](X[key])
                else:
                    buckets[key] = list(X[key].unique())
     
        return buckets
    
    def split(self, X, Y, buckets, FI = None, out_logits = None):
        #heuristics = {}
        min_var = None
        min_bucket = None
        min_val = None
        
        if self.config["surrogate"]["use_FI"]:
            if self.config["surrogate"]["multi_tree"]:
        
                FI_val = FI.mean()
            else:
                
                FI_val = FICalcs[self.config["FI"]["grouping"]](FI, out_logits)
        
            
        for var in buckets.keys():
            
            #heuristics[var] = []
            #if len(buckets[var]) != 1:
                
                for bucket in buckets[var]:
                    
                    # calculate entropy 
                    
                    if type(bucket) == np.float64 or type(bucket) == np.float32 or type(bucket) == float:
                        indicies_left = X[X[var] <= bucket].index
                    else:
                        indicies_left = X[X[var] == bucket].index
                    
                    indicies_right = X.index.difference(indicies_left)

                    
                    
                    #indicies_left = np.array(indicies_left)
                    #indicies_right = np.array(X.index.difference(indicies_left))
                    
                    Y_left = Y.loc[indicies_left]
                    Y_right = Y.loc[indicies_right]
                
                
                    heuristic_left  = 0 if len(indicies_left) == 0 else splitting_functions[self.config["surrogate"]["criterion"]](Y_left)
                    heuristic_right = 0 if len(indicies_right) == 0 else splitting_functions[self.config["surrogate"]["criterion"]](Y_right)
                
                    heuristic_all = splitting_functions[self.config["surrogate"]["criterion"]](Y)
                   
                    #heuristics[var].append(val)
                    val = heuristic_all - (len(Y_left)/len(Y)*heuristic_left + len(Y_right)/len(Y)*heuristic_right)
                    if self.config["surrogate"]["use_FI"] and type(FI) != type(None):
                        #FI_val = FICalcs["Var"](FI, out_logits)
                        #val = val * (1/FI_val[var].iloc[0])
                        # WORKS:
                        #val = ALPHA*val + BETA*FI_val[var].iloc[0]
                        """DO FOR CONTRIBUTION BASED FI"""
                        if self.config["FI"]["method"] == "LRP":
                            if self.config["surrogate"]["multi_tree"]:
                                FI_left = FI.loc[indicies_left].mean()
                                FI_right =FI.loc[indicies_right].mean()
                                val = FI_left[var]*len(Y_left)/len(Y)*heuristic_left + FI_right[var]*len(Y_right)/len(Y)*heuristic_right
                            else:
                                FI_left = FICalcs[self.config["FI"]["grouping"]](FI.loc[indicies_left], out_logits.loc[indicies_left])
                                FI_right = FICalcs[self.config["FI"]["grouping"]](FI.loc[indicies_right], out_logits.loc[indicies_right])
                                val = FI_left[var].iloc[0]*len(Y_left)/len(Y)*heuristic_left + FI_right[var].iloc[0]*len(Y_right)/len(Y)*heuristic_right

                        elif self.config["FI"]["method"] == "FD":
                            if self.config["surrogate"]["multi_tree"]:
                                #val = (1/(FI_val[var] + 0.0000000001)) * val
                                #val = len(Y_left)/len(Y)*heuristic_left*1/(FI.loc[indicies_left][var].mean()+.00000001) + len(Y_right)/len(Y)*heuristic_right*1/(FI.loc[indicies_right][var].mean()+.00000001)
                                e = 0.0000000001
                                FI_all = FI_val[var] + e
                                #FI_left = FI[var].loc[indicies_left].mean() + e
                                #FI_right = FI[var].loc[indicies_right].mean() + e
                                
                                val = (FI_all)*(heuristic_all - (len(Y_left)/len(Y)*heuristic_left + len(Y_right)/len(Y)*heuristic_right) 
                                    #+ (1/FI_all)*(len(Y_left)/len(Y)*(FI_left) + len(Y_right)/len(Y)*(FI_right)))
                                )
                                #val =  (1/(FI_val[var] + 0.0000000001)) * (len(Y_left)/len(Y)*heuristic_left + len(Y_right)/len(Y)*heuristic_right)
                            else:
                                
                                val = (FI_val[var].iloc[0]) * val
                    #     FI_val_left = FICalcs[self.config["FI"]["grouping"]](FI.loc[indicies_left], out_logits.loc[indicies_left])
                    #     if FI_val_left[var].iloc[0] == 0:
                    #         FI_val_left[var].iloc[0] = .00001
                    #     FI_val_right = FICalcs[self.config["FI"]["grouping"]](FI.loc[indicies_right], out_logits.loc[indicies_right])
                    #     if FI_val_right[var].iloc[0] == 0:
                    #         FI_val_right[var].iloc[0] = .00001 
                    #     #val = val * (1/FI_val[var].iloc[0])
                    #     val = (1/FI_val_left[var].iloc[0])*len(Y_left)/len(Y)*heuristic_left + (1/FI_val_right[var].iloc[0])*len(Y_right)/len(Y)*heuristic_right
                    # else:
                    if self.config["FI"]["method"] == "FD":
                        if min_val == None or min_val < val:
                            min_val = val
                            min_var = var
                            min_bucket = bucket

                            curr_ind_left = indicies_left
                            curr_ind_right = indicies_right
                    elif self.config["FI"]["method"] == "LRP":
                        if min_val == None or min_val > val:
                            min_val = val
                            min_var = var
                            min_bucket = bucket

                            curr_ind_left = indicies_left
                            curr_ind_right = indicies_right
                    
            #else:

        return (min_var, min_bucket, min_val, curr_ind_left, curr_ind_right)

    
    def _forward(self, Val):
        if self.is_leaf:
            return self.return_value
        
        if type(self.val_bucket) == float or type(self.val_bucket) == np.float64:
            if Val[self.feature_index] <= self.val_bucket:
                return self.left_node._forward(Val)
            else:
                return self.right_node._forward(Val)
        else:
            if Val[self.feature_index] == self.val_bucket:
                return self.left_node._forward(Val)
            else:
                return self.right_node._forward(Val)
    
    def printer(self):
        if self.is_leaf:
            return str(self.return_value)
        else:
            obj_to_Print = {"Feature": self.feature_index, "Value": self.val_bucket}
            return str(obj_to_Print) +  "\n" + "left: "+ self.left_node.printer() + "    right:"   + self.right_node.printer()
            
    def self_print(self):
        obj_to_Print = {"Feature": self.feature_index, "Value": self.val_bucket}
        print(obj_to_Print)
    
    def get_parent(self):
        return self.parent_node
    










    
