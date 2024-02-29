"""This file describes all functions which can be used as cause-effect functions in an SCM"""
# import torch
import numpy as np
from typing import List
import random

# -------------------- You can define your data-generating functions below (see readme) --------------------------------


# ------------ Some pre-defined functions. Additional functions for a specific use case can be added -------------------
def f_linear(parents: List[str]):
    """
    Creates a linear combination of the parent's values with weights drawn from U(0.5, 2) and default value 0
    """
    weights = {p: random.uniform(0.5, 2.0) for p in parents}
    default_value = 0.0

    def f(**kwargs):
        if len(kwargs) == 0:
            mu = default_value
        else:
            mu = 0.0

        for p in parents:
            mu += weights[p] * kwargs[p]
        return mu

    return f


def f_linear_noise(parents: List[str]):
    """
    Creates a linear combination of the parent's values with weights drawn from U(0.5, 2) and default value 0
    + noise drawn from N(0, 0.5) at every sampling
    """
    weights = {p: random.uniform(0.5, 2.0) for p in parents}
    default_value = 0.0

    def f(**kwargs):
        if len(kwargs) == 0:
            mu = default_value
        else:
            mu = 0.0

        for p in parents:
            mu += weights[p] * kwargs[p]
        return mu + random.gauss(0, 0.5)

    return f


def f_interaction(parents: List[str]):
    """
    Creates a linear combination of the parent's values with all weights = 1 and adds the multiplication of two randomly
    chosen parents.
    """
    if len(parents) != 0:
        interact_par1 = random.choice(parents)
        interact_par2 = random.choice(parents)

    def interaction(**kwargs):
        mu = 0
        for p in parents:
            mu += kwargs[p]
        if len(parents) != 0:
            mu += kwargs[interact_par1] * kwargs[interact_par2]
        return mu
    return interaction


def f_or(parents: List[str]):
    """
    Creates a boolean OR function over the causes/parents of a variable.
    """

    def f(**kwargs):
        res = False
        for p in parents:
            res = res or kwargs[p]
        return res

    return f


def f_and(parents: List[str]):
    """
    Creates a boolean AND function over the causes/parents of a variable with default value false.
    """
    def f(**kwargs):
        res = None
        for p in parents:
            if not res:
                res = int(kwargs[p])
            else:
                res = res and int(kwargs[p])
        if not res:
            res = 0  # default value
        return res

    return f

#
# class RandFct(torch.nn.Module):
#     """Definition of a neural network with random weights needed for the "f_rand_net" function"""
#     def __init__(self, n: int, n_hidden_nodes: int, n_hidden_layers: int):
#         """Initialize the network with n input parameters and n_hidden_layers with n_hidden nodes."""
#         super(RandFct, self).__init__()
#         self.inlayer = torch.nn.Linear(n, n_hidden_nodes)
#         self.hiddens = []
#         self.n_hidden_layers = n_hidden_layers
#         [self.hiddens.append(torch.nn.Linear(n_hidden_nodes, n_hidden_nodes)) for _ in range(self.n_hidden_layers)]
#         self.outlayer = torch.nn.Linear(n_hidden_nodes, 1)
#
#         for i in range(self.n_hidden_layers):
#             self.hiddens[i].weight.data.uniform_(-1, 1)
#             self.hiddens[i].bias.data.uniform_(-20, 20)
#
#     def forward(self, x):
#         x = torch.relu(self.inlayer(x))
#         for i in range(self.n_hidden_layers):
#             x = torch.relu(self.hiddens[i](x))
#         return self.outlayer(x) + random.gauss(0, 1)
#
#
# def f_rand_net(parents: List[str]):
#     """
#     Create a random function by initializing an MLP with random weights.
#     """
#     net = RandFct(len(parents), 10, 5)
#
#     def f(**kwargs):
#         # collect parents values
#         vals = torch.tensor([kwargs[p] for p in parents]).float()
#         out = net(vals)
#         return float(out)
#     return f


def f_discrete2_random(parents: List[str]):
    """
    Creates a discrete random function returning a binary value per possible parent value combination
    """
    # create table of size 2^n where n is the number of parents
    table = np.random.choice([0, 1], size=(2,)*len(parents))

    def discrete2_random(**kwargs):
        indices = tuple([kwargs[p] for p in parents])
        # give the value of the table representing the function value and squash it between 0 and 1
        return table[indices]

    return discrete2_random

