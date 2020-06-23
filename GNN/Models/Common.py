from math import sqrt

import torch
from torch.nn import functional as tfunc
from torch.nn.parameter import Parameter
from Utils import DEVICE
import numpy as np


#############################
### 0 : Utility functions ###
#############################


# Extracting the input elements (x_indices, edge_index, edge_type) from the padded tensor in the batch
def unpack_to_input_tpl(in_tensor, grapharea_size, max_edges):
    x_indices = in_tensor[(in_tensor[0:grapharea_size] != -1).nonzero().flatten()]
        # shortcut for the case when there is no sense
    if x_indices.nonzero().shape[0] == 0:
        edge_index = torch.zeros(size=(2,max_edges)).to(DEVICE)
        edge_type = torch.zeros(size=(max_edges,)).to(DEVICE)
        return (x_indices, edge_index, edge_type)
    edge_sources_indices = list(map(lambda idx: idx + grapharea_size,
                                    [(in_tensor[grapharea_size:grapharea_size + max_edges] != -1).nonzero().flatten()]))
    edge_sources = in_tensor[edge_sources_indices]
    edge_destinations_indices = list(map(lambda idx: idx + grapharea_size + max_edges,
                                         [(in_tensor[
                                           grapharea_size + max_edges:grapharea_size + 2 * max_edges] != -1).nonzero().flatten()]))
    edge_destinations = in_tensor[edge_destinations_indices]
    edge_type_indices = list(map(lambda idx: idx + grapharea_size + 2 * max_edges,
                                 [(in_tensor[grapharea_size + 2 * max_edges:] != -1).nonzero().flatten()]))
    edge_type = in_tensor[edge_type_indices]

    edge_index = torch.stack([edge_sources, edge_destinations], dim=0)

    return (x_indices, edge_index, edge_type)

# splitting into the 2 parts, globals and senses
def unpack_input_tensor(in_tensor, grapharea_size):

    max_edges = int(grapharea_size**1.5)
    if len(in_tensor.shape) > 1:
        in_tensor = in_tensor.squeeze()

    in_tensor_globals, in_tensor_senses = torch.split(in_tensor, split_size_or_sections=in_tensor.shape[0]//2, dim=0)
    (x_indices_g, edge_index_g, edge_type_g) = unpack_to_input_tpl(in_tensor_globals, grapharea_size, max_edges)
    (x_indices_s, edge_index_s, edge_type_s) = unpack_to_input_tpl(in_tensor_senses, grapharea_size, max_edges)
    return ((x_indices_g, edge_index_g, edge_type_g), (x_indices_s, edge_index_s, edge_type_s))


###################################
### 1: Self-attention mechanism ###
###################################

class SelfAttention(torch.nn.Module):
    # if operating with multiple heads, I use concatenation
    def __init__(self, dim_input_context, dim_input_elems, dim_qkv, num_multiheads):
        super(SelfAttention, self).__init__()
        self.d_input_context = dim_input_context
        self.d_input_elems = dim_input_elems
        self.d_qkv = dim_qkv  # the dimensionality of queries, keys and values - down from self.d_input
        self.num_multiheads = num_multiheads


        self.Wq_ls = torch.nn.ModuleList([torch.nn.Linear(in_features=self.d_input_context,
                                                          out_features=self.d_qkv, bias=False)
                                          for _current_head in range(self.num_multiheads)])
        self.Wk_ls = torch.nn.ModuleList([torch.nn.Linear(in_features=self.d_input_elems,
                                                          out_features=self.d_qkv, bias=False)
                                          for _current_head in range(self.num_multiheads)])
        self.Wv_ls = torch.nn.ModuleList([torch.nn.Linear(in_features=self.d_input_elems,
                                                          out_features=self.d_qkv, bias=False)
                                          for _current_head in range(self.num_multiheads)])


    def forward(self, input_q, input_kv, k):
        results_of_heads = []
        for current_head in range(self.num_multiheads):
            # Self-attention:
            input_query = input_q#.repeat(self.num_multiheads, 1)
            query = self.Wq_ls[current_head](input_query)

            # <= k keys, obtained projecting the embeddings of the selected senses
            #input_kv = torch.nn.functional.pad(input_kv, [0, 0, 0, k - input_kv.shape[0]])
            #input_keysandvalues = input_kv#.repeat(self.num_multiheads, 1)
            keys = self.Wk_ls[current_head](input_kv)

            # Formula for self-attention scores: softmax{(query*key)/sqrt(d_k)}
            selfatt_logits_0 = torch.matmul(query, keys.t()).squeeze()[0:keys.shape[0]]
            selfatt_logits_1 = selfatt_logits_0 / sqrt(self.d_qkv)

            # n: we want to operate in chunks if we are in a multi-head setting
            selfatt_scores = tfunc.softmax(selfatt_logits_1, dim=0)

            # Weighted sum: Σ(score*value)
            values = self.Wv_ls[current_head](input_kv)
            result_elems = values*selfatt_scores.unsqueeze(dim=1)

            result_sum = torch.sum(result_elems, dim=0)
            results_of_heads.append(result_sum)

        return torch.cat(results_of_heads, dim=0)

#############################################
### 2: Initialize common model parameters ###
#############################################

def init_model_parameters(model, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_wordlist,
                          include_globalnode_input, include_sensenode_input, predict_senses,
                          batch_size, n_layers, n_hid_units, dropout_p):
    model.grapharea_matrix = grapharea_matrix
    model.vocabulary_wordlist = vocabulary_wordlist
    model.include_globalnode_input = include_globalnode_input
    model.include_sensenode_input = include_sensenode_input
    model.predict_senses = predict_senses
    model.last_idx_senses = graph_dataobj.node_types.tolist().index(1)
    model.last_idx_globals = graph_dataobj.node_types.tolist().index(2)
    model.grapharea_size = grapharea_size
    model.dim_embs = graph_dataobj.x.shape[1]
    model.batch_size = batch_size
    model.n_layers = n_layers
    model.hidden_size = n_hid_units
    model.dropout = torch.nn.Dropout(p=dropout_p)
    return




######################
### 3: DropConnect ###
######################


# dropout_module = torch.nn.Dropout(p=0.5, inplace=False); dropout_module(some_input):
# During training, randomly zeroes some of the elements of the input tensor with probability p (Bernoulli distribution).
# This function is an alternative version of: torchnlp.nn.weight_drop._weight_drop(...)
def weight_drop(module, weights_names_ls, dropout_p):

    original_module_forward = module.forward
    forward_with_drop = ForwardWithDrop(weights_names_ls, module, dropout_p, original_module_forward)
    setattr(module, 'forward', forward_with_drop)
    return module

# Functions are only pickle-able if they are defined at the top-level of a module,
# so we create a class that is first initialized and then called as the forward()
class ForwardWithDrop(object):
    def __init__(self,weights_names_ls, module, dropout_p, original_module_forward):
        self.weights_names_ls = weights_names_ls
        self.module = module
        self.dropout_p = dropout_p
        self.original_module_forward = original_module_forward

    def __call__(self, *args, **kwargs): # the function formerly known as "forward_new"
        for name_param in self.weights_names_ls:
            param = self.module._parameters.get(name_param)
            param_with_droput = Parameter(torch.nn.functional.dropout(param, p=self.dropout_p, training=self.module.training),
                                          requires_grad=param.requires_grad)
            self.module._parameters.__setitem__(name_param, param_with_droput)

        return self.original_module_forward(*args, **kwargs)





