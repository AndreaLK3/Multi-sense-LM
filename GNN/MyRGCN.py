import torch
from torch_geometric.nn import RGCNConv
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tfunc
# import torch.nn.modules.batchnorm as batchnorm
# import Graph.DefineGraph as DG
# import GNN.SenseLabeledCorpus as SLC
# import sqlite3
# import os
# import pandas as pd
# from math import inf
# import Graph.Adjacencies as AD
# import numpy as np
# from time import time
# from Utils import DEVICE
# import GNN.DataLoading as DL
# import GNN.ExplorePredictions as EP
# import math
# from torch.nn.parameter import Parameter
# from torch.nn.modules.module import Module

### RGCN, using the torch-geometric rgcn-conv implementation. Currently, it has:
###     1 RGCN layer that operates on the selected area of the the graph
###     2 linear layers, that go from the RGCN representation to the global classes and the senses' classes
class PremadeRGCN(torch.nn.Module):
    def __init__(self, data):
        super(PremadeRGCN, self).__init__()
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.conv1 = RGCNConv(in_channels=data.x.shape[1], # doc: "Size of each input sample " in the example, num_nodes
                              out_channels=data.x.shape[1], # doc: "Size of each output sample "
                              num_relations=data.num_relations,
                              num_bases=data.num_relations)
        self.linear2global = torch.nn.Linear(in_features=data.x.shape[1],
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        self.linear2sense = torch.nn.Linear(in_features=data.x.shape[1], out_features=self.last_idx_senses, bias=True)

    def forward(self, batchinput_ls):  # given the batches, the current node is at index 0
        predictions_globals_ls = []
        predictions_senses_ls = []
        for (x, edge_index, edge_type) in batchinput_ls:
            rgcn_conv = self.conv1(x, edge_index, edge_type)
            # normalizer = batchnorm.BatchNorm1d(num_features=x.shape[1])
            # normalized_rgcn_conv = normalizer(rgcn_conv)
            x_Lplus1 = tfunc.relu(rgcn_conv)
            x1_current_node = x_Lplus1[0]  # current_node_index
            logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
            logits_sense = self.linear2sense(x1_current_node)

            sample_predictions_globals = tfunc.log_softmax(logits_global, dim=0)
            predictions_globals_ls.append(sample_predictions_globals)
            sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
            predictions_senses_ls.append(sample_predictions_senses)

        return torch.stack(predictions_globals_ls, dim=0), torch.stack(predictions_senses_ls, dim=0)


######## Functions for manual GCN convolution, and Rel-GCN
# b_L : N x d
def gcn_convolution(H, A, W_L, b_L):
    support = torch.mm(H, W_L)
    gcn_conv_result = torch.mm(A, support) + b_L
    return gcn_conv_result


def rgcn_convolution(H, Ar_ls, W_all):
    N = H.shape[0]
    d = H.shape[1]

    # Ar_all is the list of adjacency matrices for the different kinds of edges (/subgraphs).
    # we add here the 0-th, that will be used for W_0^l * h_i^l
    A0 = torch.diag(torch.ones(size=(H.shape[0],)))

    X = [A0] + Ar_ls
    Ar_all = torch.stack([A0] + Ar_ls) # prepend

    b_L = torch.zeros((N,d)) # we have no bias in our convolutions for now. We may add it as a learnable parameter.

    self_connection = gcn_convolution(H, Ar_all[0], W_all[0], b_L)
    
    sum = 0
    for r in range(1, Ar_all.shape[0]):
        Ar = Ar_all[r]
        Wr = W_all[r]
        rel_contribution = gcn_convolution(H, Ar, Wr, b_L)
        # Normalizing constant: cardinality of the neighbourhood (if > 1)
        c_ir_s = []
        for i in range(N):
            c_ir = [j for j in range(N) if Ar[i][j] != 0].__len__()
            c_ir_s.append([max(c_ir,1) for col in range(d)])
        c_ir_s = torch.tensor(c_ir_s)
        sum = sum + rel_contribution / c_ir_s

    sum = sum + self_connection

    return sum
######

######## Tools to split the input of the forward call, (x, edge_index, edge_type),
######## into subgraphs using different adjacency matrices.

grapharea_size = 32
d = 10
edge_index = torch.tensor([[1,20,11,15, 0,4,8, 12,30,28,26,21] , [0,7,4,5, 3,6,9, 12,5,15,25,31]])
edge_type = torch.tensor([0,0,0,0, 1,1,1, 2,2,2,2,2])
x = torch.rand(size=(grapharea_size, d))

def split_edge_index(edge_index, edge_type):

    sections_cutoffs = [i for i in range(edge_type.shape[0]-1) if edge_type[i] != edge_type[i-1]] + [edge_type.shape[0]]
    sections_lengths = [sections_cutoffs[i+1] - sections_cutoffs[i] for i in range(len(sections_cutoffs)-1)]

    split_sources = torch.split(edge_index[0], sections_lengths)
    split_destinations = torch.split(edge_index[1], sections_lengths)

    return (split_sources, split_destinations)


def get_adj_matrix(sources, destinations, grapharea_size):
    A = torch.zeros(size=(grapharea_size, grapharea_size))

    for e in range(sources.shape[0]):
        i = sources[e]
        j = destinations[e]

        A[i][j]=1
    return A


def create_adj_matrices(x, edge_index, edge_type):
    grapharea_size = x.shape[0]

    (split_sources, split_destinations) = split_edge_index(edge_index, edge_type)
    A_ls = []
    for seg in range(len(split_sources)):
        sources = split_sources[seg]
        destinations = split_destinations[seg]
        A_ls.append(get_adj_matrix(sources, destinations, grapharea_size))

    return A_ls
######

### RGCN, using my implementation. Splits the forward()input into different adjacency matrices.
### The weights matrices are defined manually, with no basis decomposition. The structure is equivalent to PremadeRGCN:
###     1 RGCN layer that operates on the selected area of the the graph
###     2 linear layers, that go from the RGCN representation to the global classes and the senses' classes
class MyNetRGCN(torch.nn.Module):
    def __init__(self, data):
        super(MyNetRGCN, self).__init__()
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.d = data.x.shape[1]

        # Weights matrices, Wr. W0 for the previous layer's connection, the others for the relations in R
        self.Wr_ls = []
        for r in range(data.num_relations+1):
            W_r = torch.empty(size=(self.d,self.d))
            torch.nn.init.normal_(W_r, mean=0.0, std=1.0)
            self.Wr_ls.append(W_r)

        # 2nd part of the network as before: 2 linear layers from the RGCN representation to the logits
        self.linear2global = torch.nn.Linear(in_features=self.d,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        self.linear2sense = torch.nn.Linear(in_features=self.d, out_features=self.last_idx_senses, bias=True)

    def forward(self, batchinput_ls):  # given the batches, the current node is at index 0
        predictions_globals_ls = []
        predictions_senses_ls = []
        for (x, edge_index, edge_type) in batchinput_ls:

            Ar_ls = (create_adj_matrices(x, edge_index, edge_type))
            # rgcn_conv = self.conv1(x, edge_index, edge_type)
            rgcn_conv_rep = rgcn_convolution(x, Ar_ls, self.Wr_ls)

            x_Lplus1 = tfunc.relu(rgcn_conv_rep)

            x1_current_node = x_Lplus1[0]  # current_node_index
            logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
            logits_sense = self.linear2sense(x1_current_node)

            sample_predictions_globals = tfunc.log_softmax(logits_global, dim=0)
            predictions_globals_ls.append(sample_predictions_globals)
            sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
            predictions_senses_ls.append(sample_predictions_senses)

        return torch.stack(predictions_globals_ls, dim=0), torch.stack(predictions_senses_ls, dim=0)
