import torch
from torch_geometric.nn import RGCNConv, GCNConv, GatedGraphConv, GATConv
import Utils
import Filesystem as F
import numpy as np
import logging
import torch.nn.functional as tfunc
from time import time
from Utils import DEVICE, MAX_EDGES_PACKED
from torch.nn.parameter import Parameter
import Graph.GraphArea as GA
from math import sqrt, inf

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


    def forward(self, input_q, input_kv):
        # Self-attention:
        input_query = input_q.repeat(self.num_multiheads, 1)
        query = self.Wq(input_query)
        # <= k keys, obtained projecting the embeddings of the selected senses
        input_kv = torch.nn.functional.pad(input_kv, [0, 0, 0, self.k - input_kv.shape[0]])
        input_keysandvalues = input_kv.repeat(self.num_multiheads, 1)
        keys = self.Wk(input_keysandvalues)
        # Formula for self-attention scores: softmax{(query*key)/sqrt(d_k)}
        selfatt_logits_0 = torch.matmul(query, keys.t()).squeeze()[0:keys.shape[0]]
        selfatt_logits_1 = selfatt_logits_0 / sqrt(self.d_qkv)
        # n: we want to operate in chunks if we are in a multi-head setting
        selfatt_scores = tfunc.softmax(selfatt_logits_1, dim=0)
        # Weighted sum: Σ(score*value)
        values = self.Wv(input_keysandvalues)
        result_elems = selfatt_scores*values
        result_sum = torch.sum(result_elems, dim=0)

        return result_sum





####################
### 2: GRU + GAT ###
####################
class GRU_GAT(torch.nn.Module):
    def __init__(self, data, grapharea_size, num_attention_heads, include_senses):
        super(GRU_GAT, self).__init__()
        self.data = data
        self.include_senses = include_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]

        self.h1_state_dim = 2 * self.d
        self.h2_state_dim = self.d

        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_node = Parameter(torch.tensor([0]), requires_grad=False)
        self.nodestate_zeros = Parameter(torch.zeros(size=(1,self.d)), requires_grad=False)

        # Input signals: current global’s word embedding || global’s node-state (|| sense’s node state)
        self.concatenated_input_dim = 2 * self.d if not (self.include_senses) else 3 * self.d

        # GAT
        self.gat_globals = GATConv(in_channels=self.d,
                                   out_channels=self.d // num_attention_heads, heads=num_attention_heads, concat=True,
                                   negative_slope=0.2, dropout=0, bias=True)
        if self.include_senses:
            self.gat_senses = GATConv(in_channels=self.d,
                                   out_channels=self.d // num_attention_heads, heads=num_attention_heads, concat=True,
                                   negative_slope=0.2, dropout=0, bias=True)
        # self.gat_out_channels = self.d // (num_attention_heads // 2)


        # GRU: we update these memory buffers manually, there is no gradient. Set as a Parameter to DataParallel-ize it
        self.memory_h1 = Parameter(torch.zeros(size=(1, self.h1_state_dim)), requires_grad=False)
        self.memory_h2 = Parameter(torch.zeros(size=(1, self.h2_state_dim)), requires_grad=False)

        # GRU: 1st layer
        self.U_z_1 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h1_state_dim, bias=False)
        self.W_z_1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim, bias=False)
        self.U_r_1 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h1_state_dim, bias=False)
        self.W_r_1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim, bias=False)

        self.U_1 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h1_state_dim, bias=True)
        self.W_1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim, bias=True)
        self.dropout = torch.nn.Dropout(p=0.1)

        # GRU: 2nd layer
        self.U_z_2 = torch.nn.Linear(in_features=self.h2_state_dim, out_features=self.h2_state_dim, bias=False)
        self.W_z_2 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h2_state_dim, bias=False)
        self.U_r_2 = torch.nn.Linear(in_features=self.h2_state_dim, out_features=self.h2_state_dim, bias=False)
        self.W_r_2 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h2_state_dim, bias=False)

        self.U_2 = torch.nn.Linear(in_features=self.h2_state_dim, out_features=self.h2_state_dim, bias=True)
        self.W_2 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h2_state_dim, bias=True)


        # 2nd part of the network: globals' logits and senses' self-attention prediction
        self.linear2global = torch.nn.Linear(in_features=self.h2_state_dim,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)

        if self.include_senses:
            self.k = 500 # the number of "likely globals". We choose among <=k senses
            self.likely_senses_embs = Parameter(-1*torch.ones(size=(self.k, self.d)), requires_grad=False)
            self.d_qkv = 150 # the dimensionality of queries, keys and values - down from self.d(embeddings)
            self.mySelfAttention = SelfAttention(dim_input_context=self.concatenated_input_dim, dim_input_elems=self.d,
                                                 dim_qkv=self.d_qkv, num_multiheads=1)



    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0

        predictions_globals_ls = []
        predictions_senses_ls = []
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_h1.detach_()
        self.memory_h2.detach_()

        if batchinput_tensor.shape[0] > 1:
            sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
        else:
            sequences_in_the_batch_ls = [batchinput_tensor]

        for padded_sequence in sequences_in_the_batch_ls:
            padded_sequence = padded_sequence.squeeze() #thus we have torch.Size([35,544])
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

            for (x_indices, edge_index, edge_type), (x_indices_s, edge_index_s, edge_type_s) in sequence_lts:

                # Input signal n.1: the current (global) word
                currentword_embedding = self.X.index_select(dim=0, index=x_indices[0])

                # Input signal n.2: the node-state of the word in the KB-graph, obtained applying the GNN
                x = self.X.index_select(dim=0, index=x_indices.squeeze())
                x_attention_state = self.gat_globals(x, edge_index)
                currentword_node_state = x_attention_state.index_select(dim=0, index=self.select_first_node)

                # Input signal n.3: the node-state of the current sense; + concatenating the input signals
                if self.include_senses:
                    if x_indices_s.nonzero().shape[0] == 0: # no sense was specified
                        currentsense_node_state = self.nodestate_zeros
                    else: # sense was specified
                        x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
                        x_attention_state_s = self.gat_senses(x_s, edge_index_s)
                        currentsense_node_state = x_attention_state_s.index_select(dim=0,index=self.select_first_node)
                    input_signals = torch.cat([currentword_embedding, currentword_node_state, currentsense_node_state], dim=1)
                else:
                    input_signals = torch.cat([currentword_embedding, currentword_node_state], dim=1)

                # GRU: Layer 1
                z_1 = torch.sigmoid(self.W_z_1(input_signals) + self.U_z_1(self.memory_h1))
                r_1 = torch.sigmoid(self.W_r_1(input_signals) + self.U_r_1(self.memory_h1))
                h_tilde_1 = torch.tanh(self.dropout(self.W_1(input_signals)) + self.U_1(r_1 * self.memory_h1))
                h1 = z_1 * h_tilde_1 + (torch.tensor(1)-z_1) * self.memory_h1

                self.memory_h1.data.copy_(h1.clone().detach()) # store h in memory

                # GRU: Layer 2  - globals task
                z_2 = torch.sigmoid(self.W_z_2(h1) + self.U_z_2(self.memory_h2))
                r_2 = torch.sigmoid(self.W_r_2(h1) + self.U_r_2(self.memory_h2))
                h_tilde_2 = torch.tanh(self.dropout(self.W_2(h1)) + self.U_2(r_2 * self.memory_h2))
                h2 = z_2 * h_tilde_2 + (torch.tensor(1) - z_2) * self.memory_h2

                self.memory_h2.data.copy_(h2.clone().detach())  # store h in memory


                # 2nd part of the architecture: predictions

                # Globals
                logits_global = self.linear2global(h2)
                sample_predictions_globals = tfunc.log_softmax(logits_global, dim=1)
                predictions_globals_ls.append(sample_predictions_globals)

                # Senses
                if self.include_senses:
                    most_likely_globals = torch.sort(logits_global, descending=True)[1] \
                                                    [0][0:self.k] + self.last_idx_senses
                    # for every one of the most likely globals, retrieve the neighbours,
                    neighbours_indices_ls = [node_idx for neighbours_subls in list(map(
                        lambda global_node_idx:
                            GA.get_indices_area_toinclude(self.data.edge_index, self.data.edge_type,
                                                          global_node_idx.cpu().item(), area_size=32, max_hops=1)[0],
                        most_likely_globals))
                    for node_idx in neighbours_subls]
                    # and keep only their neighbours that are (in the range of) sense nodes
                    likely_senses_indices = torch.tensor(list(filter(
                        lambda node_idx : node_idx in range(0, self.last_idx_senses),
                        neighbours_indices_ls)))[0:self.k]
                    if torch.cuda.is_available():
                        likely_senses_indices = likely_senses_indices.to("cuda:"+str(torch.cuda.current_device()))
                    self.likely_senses_embs.data = self.X.index_select(dim=0, index=likely_senses_indices)
                    # Self-attention:
                    # One query: the context, taken from a no-gradient copy of h1:
                    query = torch.matmul(self.memory_h2, self.Wq)
                    # <= k keys, obtained projecting the embeddings of the selected senses
                    keys = torch.matmul(self.likely_senses_embs, self.Wk)
                    keys_padded = torch.nn.functional.pad(keys, [0, 0, 0, self.k - keys.shape[0]])
                    # Formula for self-attention scores: softmax{(query*key)/sqrt(d_k)}
                    selfatt_logits_0 = torch.matmul(query, keys_padded.t()).squeeze()[0:keys.shape[0]]
                    selfatt_logits_1 = selfatt_logits_0 / sqrt(self.d_qkv)
                    selfatt_scores = tfunc.log_softmax(selfatt_logits_1, dim=0)
                    # We have a probability distribution. Assign probabilities to the selected senses, the rest are ==0.
                    sample_predictions_senses = torch.ones(size=(self.last_idx_senses,))*(-100)
                    if torch.cuda.is_available():
                        sample_predictions_senses = sample_predictions_senses.to("cuda:"+str(torch.cuda.current_device()))
                    sample_predictions_senses[likely_senses_indices]=selfatt_scores.squeeze()
                    predictions_senses_ls.append(sample_predictions_senses)

                else:
                    predictions_senses_ls.append(torch.tensor(0).to(DEVICE)) # so I don't have to change the interface elsewhere

        return torch.stack(predictions_globals_ls, dim=0).squeeze(), \
               torch.stack(predictions_senses_ls, dim=0).squeeze()
