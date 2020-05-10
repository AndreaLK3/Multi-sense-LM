from math import sqrt

import torch
from torch.nn import functional as tfunc
from torch.nn.parameter import Parameter
from Utils import DEVICE
import numpy as np


#############################
### 0 : Utility functions ###
#############################

# Tools to split the input of the forward call, (x, edge_index, edge_type),
# into subgraphs (that can use different adjacency matrices).
def split_edge_index(edge_index, edge_type):
    sections_cutoffs = [i for i in range(edge_type.shape[0]) if edge_type[i] != edge_type[i-1]] + [edge_type.shape[0]]
    if 0 not in sections_cutoffs:
        sections_cutoffs = [0] + sections_cutoffs # prepend, to deal with the case of 1 edge
    sections_lengths = [sections_cutoffs[i+1] - sections_cutoffs[i] for i in range(len(sections_cutoffs)-1)]
    split_sources = torch.split(edge_index[0], sections_lengths)
    split_destinations = torch.split(edge_index[1], sections_lengths)

    return (split_sources, split_destinations)


def get_antonym_nodes(edge_index, edge_type, antonym_edge_number):
    _sources = edge_index[0].masked_select(torch.eq(edge_type, antonym_edge_number))
    destinations = edge_index[1].masked_select(torch.eq(edge_type, antonym_edge_number))
    return destinations

######

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
    print("in_tensor.shape=" + str(in_tensor.shape))
    max_edges = int(grapharea_size**1.5)
    in_tensor = in_tensor.squeeze()
    print("in_tensor.shape[0]//2=" + str(in_tensor.shape[0]//2))
    in_tensor_globals, in_tensor_senses = torch.split(in_tensor, split_size_or_sections=in_tensor.shape[0]//2, dim=0)
    (x_indices_g, edge_index_g, edge_type_g) = unpack_to_input_tpl(in_tensor_globals, grapharea_size, max_edges)
    (x_indices_s, edge_index_s, edge_type_s) = unpack_to_input_tpl(in_tensor_senses, grapharea_size, max_edges)
    return ((x_indices_g, edge_index_g, edge_type_g), (x_indices_s, edge_index_s, edge_type_s))

# numpy version
def unpack_to_input_tpl_numpy(in_ndarray, grapharea_size, max_edges):
    CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'gpu:' + str(torch.cuda.current_device())
    x_indices = np.extract(condition=[elem != -1 for elem in in_ndarray[0:grapharea_size]], arr=in_ndarray[0:grapharea_size])
        # shortcut for the case when there is no sense
    if len(x_indices) == 0:
        edge_index = torch.zeros(size=(2,max_edges)).to(CURRENT_DEVICE)
        edge_type = torch.zeros(size=(max_edges,)).to(CURRENT_DEVICE)
        return (x_indices, edge_index, edge_type)
    edge_sources = np.extract(condition=[elem!=-1 for elem in in_ndarray[grapharea_size:grapharea_size + max_edges]],
                              arr=in_ndarray[grapharea_size:grapharea_size + max_edges])
    edge_destinations = np.extract(condition=[elem!=-1 for elem in in_ndarray[grapharea_size + max_edges:grapharea_size + 2 * max_edges]],
                              arr=in_ndarray[grapharea_size + max_edges:grapharea_size + 2 * max_edges])
    edge_type = torch.tensor(np.extract(condition=[elem!=-1 for elem in in_ndarray[grapharea_size + 2 * max_edges:]],
                              arr=in_ndarray[grapharea_size + 2 * max_edges:])).to(CURRENT_DEVICE)

    edge_index = torch.stack([torch.tensor(edge_sources).to(CURRENT_DEVICE),
                              torch.tensor(edge_destinations).to(CURRENT_DEVICE)], dim=0)

    return (x_indices, edge_index, edge_type)

def unpack_input_tensor_numpy(batchinput_ndarray, grapharea_size):
    print("batchinput_ndarray.shape=" + str(batchinput_ndarray.shape))
    max_edges = int(grapharea_size**1.5)
    print("batchinput_ndarray.shape[0]//2=" + str(batchinput_ndarray.shape[0]//2))
    in_tensor_globals, in_tensor_senses = np.split(batchinput_ndarray, indices_or_sections=[batchinput_ndarray.shape[0]//2], axis=0)
    (x_indices_g, edge_index_g, edge_type_g) = unpack_to_input_tpl_numpy(in_tensor_globals, grapharea_size, max_edges)
    (x_indices_s, edge_index_s, edge_type_s) = unpack_to_input_tpl_numpy(in_tensor_senses, grapharea_size, max_edges)
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


######################
### 2: DropConnect ###
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

