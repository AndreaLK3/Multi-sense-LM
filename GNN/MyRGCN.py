import torch
from torch_geometric.nn import RGCNConv
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tfunc
import torch.nn.modules.batchnorm as batchnorm
import Graph.DefineGraph as DG
import GNN.SenseLabeledCorpus as SLC
import sqlite3
import os
import pandas as pd
from math import inf
import Graph.Adjacencies as AD
import numpy as np
from time import time
from Utils import DEVICE
import GNN.DataLoading as DL
import GNN.ExplorePredictions as EP
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

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


########
# b_L : N x d
def gcn_convolution(H, A, W_L, b_L):
    support = torch.mm(H, W_L)
    gcn_conv_result = torch.mm(A, support) + b_L
    return gcn_conv_result


d = 2
N = 3
H = torch.tensor([[5,10], [0.1,0.2], [-1,-1]])
W_all= torch.stack([torch.ones((d,d)) * 2, torch.ones((d,d)) * 10])
Ar_all = torch.tensor([[ [0,1,1] , [0,0,1] , [1,0,0] ], [ [0,0,0] , [1,0,0] , [0,1,0] ]]).to(torch.float)
def rgcn_convolution(H, Ar_all, W_all):
    # Ar_s is a list of adjacency matrices for the different kinds of edges.
    # we add here the 0-th, that will be used for W_0^l * h_i^l
    A0 = torch.diag(torch.ones(size=(H.shape[0],)))

    b_L = torch.zeros((N,d))
    self_connection = gcn_convolution(H, A0, W_all[0], b_L)
    print(self_connection.shape)

    sum = 0
    for r in range(Ar_all.shape[0]):
        Ar = Ar_all[r]
        Wr = W_all[r]
        rel_contribution = gcn_convolution(H, Ar, Wr, b_L)
        print(rel_contribution)
        sum = sum + rel_contribution

