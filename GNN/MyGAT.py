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


# ##########################################################################
# ### 1 : GRU + manually built GAT that operates on different node types ###
# ##########################################################################
# class GRU_GATN(torch.nn.Module):
#     def __init__(self, data, grapharea_size, hidden_state_dim, include_senses):
#         super(GRU_GATN, self).__init__()
#         self.include_senses = include_senses
#         self.last_idx_senses = data.node_types.tolist().index(1)
#         self.last_idx_globals = data.node_types.tolist().index(2)
#         self.N = grapharea_size
#         self.d = data.x.shape[1]
#         self.hidden_state_dim = hidden_state_dim
#         self.num_edge_types = len(torch.unique(data.edge_type))
#
#         # The embeddings matrix for: senses, globals, definitions, examples (the latter 2 may have gradient set to 0)
#         self.X = Parameter(data.x.clone().detach(), requires_grad=True)
#         self.select_first_node = Parameter(torch.tensor([0]), requires_grad=False)
#         self.node_types =Parameter(data.node_types.clone(), requires_grad=False) # Node types
#
#         # GAT: The matrices to project the different kinds of nodes
#         node_projections_matrices_ls = []
#         for node_type in range(self.num_edge_types):
#             node_projections_matrices_ls.append(torch.empty(size=(self.d, self.d)))
#         self.node_projections_matrices = Parameter(torch.stack(node_projections_matrices_ls),requires_grad=True)
#         # GAT: e_ij = A*[Wp(i)*h_i, Wp(i)*h_j]. Then, LeakyReLu to get the attention logit
#         self.A = torch.nn.Linear(in_features=2*self.d, out_features=1, bias=False)
#
#         # GRU: we update these memory buffers manually, there is no gradient. Set as a Parameter to DataParallel-ize it
#         self.memory_h1 = Parameter(torch.zeros(size=(1, self.hidden_state_dim)), requires_grad=False)
#         self.memory_h2 = Parameter(torch.zeros(size=(1, self.hidden_state_dim)), requires_grad=False)
#
#         # GRU: 1st layer
#         self.U_z_1 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
#         self.W_z_1 = torch.nn.Linear(in_features=self.d, out_features=self.hidden_state_dim, bias=False)
#         self.U_r_1 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
#         self.W_r_1 = torch.nn.Linear(in_features=self.d, out_features=self.hidden_state_dim, bias=False)
#
#         self.U_1 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=True)
#         self.W_1 = torch.nn.Linear(in_features=self.d, out_features=self.hidden_state_dim, bias=True)
#         self.dropout = torch.nn.Dropout(p=0.1)
#
#         # GRU: 2nd layer
#         self.U_z_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
#         self.W_z_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
#         self.U_r_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
#         self.W_r_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
#
#         self.U_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=True)
#         self.W_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=True)
#
#         # 2nd part of the network as before: 2 linear layers to the logits
#         self.linear2global = torch.nn.Linear(in_features=self.hidden_state_dim,
#                                              out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
#
#         if self.include_senses:
#             self.linear2sense = torch.nn.Linear(in_features=self.hidden_state_dim,
#                                                 out_features=self.last_idx_senses, bias=True)
#
#
#
#     def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
#
#         predictions_globals_ls = []
#         predictions_senses_ls = []
#         # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
#         self.memory_h1.detach_()
#         self.memory_h2.detach_()
#
#         if batchinput_tensor.shape[0] > 1:
#             sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
#         else:
#             sequences_in_the_batch_ls = [batchinput_tensor]
#
#         for padded_sequence in sequences_in_the_batch_ls:
#             padded_sequence = padded_sequence.squeeze() #thus we have torch.Size([35,544])
#             padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
#             sequence_lts = [unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]
#
#             for (x_indices, edge_index, edge_type) in sequence_lts:
#                 currentword_embedding = self.X.index_select(dim=0, index=x_indices[0])
#
#                 neighbours_x = self.X.index_select(dim=0, index=x_indices.squeeze())
#                 neighbours_nodetypes = self.node_types.index_select(dim=0, index=x_indices.squeeze())
#                 antonyms_indices = get_antonym_nodes(edge_index, edge_type, self.num_edge_types - 1)
#                 definitions = neighbours_x[torch.eq(neighbours_nodetypes,0)]
#                 examples = neighbours_x[torch.eq(neighbours_nodetypes,1)]
#                 senses = neighbours_x[torch.eq(neighbours_nodetypes,2)]
#                 globals = neighbours_x[torch.eq(neighbours_nodetypes,3)]
#
#                 #logging.info("\ncurrent_node_index = " + str(x_indices[0]))
#                 #logging.info("neighbours_indices=" + str(x_indices))
#
#                 number_of_antonyms = len(antonyms_indices)
#                 synonyms = globals[:globals.shape[0]-number_of_antonyms]
#                 if number_of_antonyms>0:
#                     antonyms = globals[-number_of_antonyms:]
#                 else:
#                     antonyms = globals[0:0]
#
#                 # GAT: Projecting different kinds of nodes
#                 projected_neighbours_ls = []
#                 neighbouring_elements = [definitions, examples, senses, synonyms, antonyms]
#
#                 for i in range(self.num_edge_types):
#                     if neighbouring_elements[i].shape[0] > 0:
#                         projected_neighbours_ls.append(torch.mm(self.node_projections_matrices[i], neighbouring_elements[i].t()))
#                 projected_neighbours = torch.cat(projected_neighbours_ls, dim=1)
#                 #logging.info("projected_neighbours.shape=" + str(projected_neighbours.shape))
#                 node_pairs = torch.cat([currentword_embedding.repeat((projected_neighbours.shape[1],1)), projected_neighbours.t()],1)
#                 # GAT: e_ij = A*[Wp(i)*h_i, Wp(i)*h_j]. Then, LeakyReLu to get the attention logit
#                 attention_logits_e = tfunc.leaky_relu(self.A(node_pairs))
#                 # GAT: softmax and weighted sum
#                 attention_coefficients_alpha = tfunc.softmax(attention_logits_e.squeeze(dim=1), dim=0)
#                 #logging.info("attention_coefficients_alpha=" + str(attention_coefficients_alpha))
#                 node_attention_state = (sum([attention_coefficients_alpha[i]*projected_neighbours[:,i] for i in range(projected_neighbours.shape[1])])).unsqueeze(0)
#
#                 # GRU: Layer 1
#                 z_1 = torch.sigmoid(self.W_z_1(node_attention_state) + self.U_z_1(self.memory_h1))
#                 r_1 = torch.sigmoid(self.W_r_1(node_attention_state) + self.U_r_1(self.memory_h1))
#                 h_tilde_1 = torch.tanh(self.dropout(self.W_1(node_attention_state)) + self.U_1(r_1 * self.memory_h1))
#                 h1 = z_1 * h_tilde_1 + (torch.tensor(1)-z_1) * self.memory_h1
#
#                 self.memory_h1.data.copy_(h1.clone()) # store h in memory
#
#                 # GRU: Layer 2
#                 z_2 = torch.sigmoid(self.W_z_2(h1) + self.U_z_2(self.memory_h2))
#                 r_2 = torch.sigmoid(self.W_r_2(h1) + self.U_r_2(self.memory_h2))
#                 h_tilde_2 = torch.tanh(self.dropout(self.W_2(h1)) + self.U_2(r_2 * self.memory_h2))
#                 h2 = z_2 * h_tilde_2 + (torch.tensor(1) - z_2) * self.memory_h2
#
#                 self.memory_h2.data.copy_(h2.clone())  # store h in memory
#
#                 # 2nd part of the architecture: predictions
#                 logits_global = self.linear2global(h2)  # shape=torch.Size([5])
#                 sample_predictions_globals = tfunc.log_softmax(logits_global, dim=1)
#                 predictions_globals_ls.append(sample_predictions_globals)
#                 if self.include_senses:
#                     logits_sense = self.linear2sense(h2)
#                     sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
#                     predictions_senses_ls.append(sample_predictions_senses)
#                 else:
#                     predictions_senses_ls.append(torch.tensor(0).to(DEVICE)) # so I don't have to change the interface elsewhere
#
#             #Utils.log_chronometer([t0,t1,t2,t3,t4,t5, t6, t7, t8])
#         return torch.stack(predictions_globals_ls, dim=0).squeeze(), \
#                torch.stack(predictions_senses_ls, dim=0).squeeze()




####################
### 2: GRU + GAT ###
####################
class GRU_GAT(torch.nn.Module):
    def __init__(self, data, grapharea_size, num_attention_heads, include_senses):
        super(GRU_GAT, self).__init__()
        self.include_senses = include_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]

        self.h1_state_dim = 2 * self.d
        self.h2_state_dim = self.d

        # The embeddings matrix for: senses, globals, definitions, examples (the latter 2 may have gradient set to 0)
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_node = Parameter(torch.tensor([0]), requires_grad=False)
        self.nodestate_zeros = Parameter(torch.zeros(size=(1,self.d)), requires_grad=False)

        # Input signals: current global’s word embedding || global’s node-state (|| sense’s node state)
        self.concatenated_input_dim = 2 * self.d if not (self.include_senses) else 3 * self.d
        # self.IH = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim, bias=True)

        # GAT
        self.gat = GATConv(in_channels=self.d,
                           out_channels=self.d // num_attention_heads, heads=num_attention_heads, concat=True, negative_slope=0.2, dropout=0, bias=True)
        #self.gat_out_channels = self.d // (num_attention_heads // 2)

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

        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=self.h2_state_dim,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)

        if self.include_senses:
            self.linear2sense = torch.nn.Linear(in_features=self.h2_state_dim,
                                                out_features=self.last_idx_senses, bias=True)



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
                x_attention_state = self.gat(x, edge_index)
                currentword_node_state = x_attention_state.index_select(dim=0, index=self.select_first_node)

                # Input signal n.3: the node-state of the current sense; + concatenating the input signals
                if self.include_senses:
                    if x_indices_s.nonzero().shape[0] == 0: # no sense was specified
                        currentsense_node_state = self.nodestate_zeros
                    else: # sense was specified
                        x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
                        x_attention_state_s = self.gat(x_s, edge_index)
                        currentsense_node_state = x_attention_state_s.index_select(dim=0,index=self.select_first_node)
                    input_signals = torch.cat([currentword_embedding, currentword_node_state, currentsense_node_state], dim=1)
                else:
                    input_signals = torch.cat([currentword_embedding, currentword_node_state], dim=1)

                # GRU: Layer 1
                z_1 = torch.sigmoid(self.W_z_1(input_signals) + self.U_z_1(self.memory_h1))
                r_1 = torch.sigmoid(self.W_r_1(input_signals) + self.U_r_1(self.memory_h1))
                h_tilde_1 = torch.tanh(self.dropout(self.W_1(input_signals)) + self.U_1(r_1 * self.memory_h1))
                h1 = z_1 * h_tilde_1 + (torch.tensor(1)-z_1) * self.memory_h1

                self.memory_h1.data.copy_(h1.clone()) # store h in memory

                # GRU: Layer 2
                z_2 = torch.sigmoid(self.W_z_2(h1) + self.U_z_2(self.memory_h2))
                r_2 = torch.sigmoid(self.W_r_2(h1) + self.U_r_2(self.memory_h2))
                h_tilde_2 = torch.tanh(self.dropout(self.W_2(h1)) + self.U_2(r_2 * self.memory_h2))
                h2 = z_2 * h_tilde_2 + (torch.tensor(1) - z_2) * self.memory_h2

                self.memory_h2.data.copy_(h2.clone())  # store h in memory

                # 2nd part of the architecture: predictions
                logits_global = self.linear2global(h2)  # shape=torch.Size([5])
                sample_predictions_globals = tfunc.log_softmax(logits_global, dim=1)
                predictions_globals_ls.append(sample_predictions_globals)
                if self.include_senses:
                    logits_sense = self.linear2sense(h2)
                    sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
                    predictions_senses_ls.append(sample_predictions_senses)
                else:
                    predictions_senses_ls.append(torch.tensor(0).to(DEVICE)) # so I don't have to change the interface elsewhere

            #Utils.log_chronometer([t0,t1,t2,t3,t4,t5, t6, t7, t8])
        return torch.stack(predictions_globals_ls, dim=0).squeeze(), \
               torch.stack(predictions_senses_ls, dim=0).squeeze()