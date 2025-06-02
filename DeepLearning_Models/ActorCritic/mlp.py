import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size):
    """
    Builds a multi-layer perceptron in Pytorch based on a user's input

    Args:
        input_size (int): the dimension of inputs to be given to the network
        output_size (int): the dimension of the output
        n_layers (int): the number of hidden layers of the network
        size (int): the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    """

    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")

    modules = []

    
    modules.append(nn.Linear(input_size, size))
    modules.append(nn.ReLU())


    for _ in range(n_layers - 1):
        modules.append(nn.Linear(size, size))
        modules.append(nn.ReLU())

   
    modules.append(nn.Linear(size, output_size))


    model = nn.Sequential(*modules)

    return model