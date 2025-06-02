import torch
from torch import nn

class TwoDWeights_LRPModel(nn.Module):
    def __init__(self, model, top_k=5):
        super().__init__()
        self.model = model
        self.top_k = top_k

        self.model.eval()
        self.layers = self._get_layers()[::-1]  # Extract only fully connected layers
        self.activations = []  # List to store activations during forward pass
    
    def _get_layers(self):
        """
        Retrieves the fully connected layers (nn.Linear) of the model.
        """
        layers = []
        # Recursively collect layers if model has nested structures
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):  # Adjust if other types like ReLU are involved
                layers.append(layer)
            elif len(list(layer.children())) > 0:
                layers.extend(self._get_layers_from_module(layer))
        return layers
    
    def _get_layers_from_module(self, module):
        """Recursively retrieves fully connected layers from nested modules."""
        layers = []
        for layer in module.children():
            if isinstance(layer, nn.Linear):
                layers.append(layer)
            elif len(list(layer.children())) > 0:
                layers.extend(self._get_layers_from_module(layer))
        return layers
    
    def _register_hooks(self):
        """
        Registers forward hooks to capture activations from each fully connected layer during the forward pass.
        """
        self.activations = []  # Reset activations list
        for layer in self.layers:
            layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """
        Hook function to capture activations from each layer.
        """
        
        self.activations.append(input[0].clone())  # Store the output of each layer
    
    def forward(self, x):
        self._register_hooks()  # Register hooks before the forward pass
        output = self.model(x)  # Standard forward pass
        return output
        
    def propagate_relevance(self, relevance, layer, activations):
        """
        Propagates the relevance scores backward using the Z-Rule.
        """
        weights = layer.weight
        weights = weights.T
        biases = layer.bias
        z = []
        CorVec = []
        
        for i in range(weights.shape[1]):
            tmp = activations*weights[:,i] + biases[i]
            z.append(tmp.sum())
            CorVec.append(tmp)
        
        z = torch.stack(z)
        #z += 1e-9 * (z >= 0).float() + (-1e-9) * (z < 0).float()
        z += 1e-9
        CorVec = torch.stack(CorVec)


        normalized_CorVec = CorVec / z.unsqueeze(1)
        
        
        relevance_scores = relevance @ normalized_CorVec
        
        return relevance_scores
    
    def backward_relevance_propagation(self, output):
        """
        Applies backward relevance propagation using captured activations and layers.
        
        Parameters:
        - output: The output from the forward pass.
        
        Returns:
        - input_relevance: Relevance scores attributed to the input layer.
        """
        relevance = output.clone()  # Start with the output layer relevance
        
        # Propagate relevance backward using the layers and activations
        # print(self.layers)
        # biases = [i.bias for i in self.layers]
        # print(self.activations)
        # print(biases)
        # print([i for i in self.model.modules()])
        vectors = []
        input_relevance = []
        with torch.no_grad():
            for i in range(len(relevance)):
                vector = torch.zeros_like(relevance)  # Create a zero vector of the same shape
                vector[i] = relevance[i]  # Set the i-th position to the corresponding relevance value
                vectors.append(vector)

            # Display the generated vectors
            for v in vectors:
                relevance = v
                for layer, activation in zip(self.layers, self.activations[::-1]):
                    relevance = self.propagate_relevance(relevance, layer, activation)
                
                input_relevance.append(relevance)
        self.activations = []
        return input_relevance


    def get_FI(self, point, printer = False):
        if printer:
            print("point", point)
            out = self.forward(point)
            print("output",out)
            rel = self.backward_relevance_propagation(out)
            print("relevence", rel)
            return out,rel
        
        out = self.forward(point)
        return out,self.backward_relevance_propagation(out)
    