from math import sqrt

import torch
from torch.nn import functional as tfunc

from Utils import DEVICE


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
    max_edges = int(grapharea_size**1.5)
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

        self.Wq = torch.nn.Linear(in_features=self.d_input_context*num_multiheads, out_features=self.d_qkv, bias=False)
        self.Wk = torch.nn.Linear(in_features=self.d_input_elems*num_multiheads, out_features=self.d_qkv, bias=False)
        self.Wv = torch.nn.Linear(in_features=self.d_input_elems*num_multiheads, out_features=self.d_qkv, bias=False)


    def forward(self, input_q, input_kv, k):
        #t0=time()
        # Self-attention:
        input_query = input_q.repeat(self.num_multiheads, 1)
        query = self.Wq(input_query)
        #t1=time()
        # <= k keys, obtained projecting the embeddings of the selected senses
        input_kv = torch.nn.functional.pad(input_kv, [0, 0, 0, k - input_kv.shape[0]])
        input_keysandvalues = input_kv.repeat(self.num_multiheads, 1)
        keys = self.Wk(input_keysandvalues)
        #t2=time()
        # Formula for self-attention scores: softmax{(query*key)/sqrt(d_k)}
        selfatt_logits_0 = torch.matmul(query, keys.t()).squeeze()[0:keys.shape[0]]
        selfatt_logits_1 = selfatt_logits_0 / sqrt(self.d_qkv)
        #t3=time()
        # n: we want to operate in chunks if we are in a multi-head setting
        selfatt_scores = tfunc.softmax(selfatt_logits_1, dim=0)
        #t4=time()
        # Weighted sum: Σ(score*value)
        values = self.Wv(input_keysandvalues)
        result_elems = values*selfatt_scores.unsqueeze(dim=1)
        #t5=time()
        result_sum = torch.sum(result_elems, dim=0)
        #t6=time()
        #logging.info("Time analysis of the SelfAttention submodule's forward()")
        #Utils.log_chronometer([t0,t1,t2,t3,t4,t5,t6])

        return result_sum