import torch
from torch_geometric.nn import RGCNConv, GCNConv, GatedGraphConv
import Utils
import Filesystem as F
import numpy as np
import logging
import torch.nn.functional as tfunc
from time import time
from Utils import DEVICE, MAX_EDGES_PACKED
from torch.nn.parameter import Parameter


######## Tools to split the input of the forward call, (x, edge_index, edge_type),
######## into subgraphs using different adjacency matrices.

def split_edge_index(edge_index, edge_type):
    # logging.info("Edge_index.shape=" + str(edge_index.shape) + " ; edge_type.shape=" + str(edge_type.shape))
    # if edge_type.shape[0] in [1, 16, 47, 85,4, 51, 58, 62, 4, 65, 38, 56]:
    #     logging.info("edge_index=" + str(edge_index))
    #     logging.info("edge_type=" + str(edge_type))
    sections_cutoffs = [i for i in range(edge_type.shape[0]) if edge_type[i] != edge_type[i-1]] + [edge_type.shape[0]]
    if 0 not in sections_cutoffs:
        sections_cutoffs = [0] + sections_cutoffs # prepend, to deal with the case of 1 edge
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


def unpack_input_tensor(in_tensor, grapharea_size, max_edges=MAX_EDGES_PACKED):
    in_tensor = in_tensor.squeeze()
    x_indices = in_tensor[(in_tensor[0:grapharea_size] != -1).nonzero().flatten()]
    # edge_sources_indices = list(map(lambda idx: idx + grapharea_size, [(in_tensor[grapharea_size:grapharea_size+max_edges] != -1).nonzero().flatten()]))
    # edge_sources = in_tensor[edge_sources_indices]
    # edge_destinations_indices = list(map(lambda idx: idx + grapharea_size + max_edges,
    #          [(in_tensor[grapharea_size+max_edges:grapharea_size+2*max_edges] != -1).nonzero().flatten()]))
    # edge_destinations = in_tensor[edge_destinations_indices]
    # edge_type_indices = list(map(lambda idx: idx + grapharea_size + 2*max_edges,
    #          [(in_tensor[grapharea_size+2*max_edges:] != -1).nonzero().flatten()]))
    # edge_type = in_tensor[edge_type_indices]
    #
    # edge_index = torch.stack([edge_sources, edge_destinations], dim=0)
    return x_indices #, edge_index, edge_type)


def unpack_bptt_elem(sequence_lts, elem_idx):

    x_indices = sequence_lts[0][elem_idx][sequence_lts[0][elem_idx]!=-1]
    edge_sources = sequence_lts[1][elem_idx][sequence_lts[1][elem_idx]!=-1]
    edge_destinations = sequence_lts[2][elem_idx][sequence_lts[2][elem_idx] != -1]
    edge_index = torch.stack([edge_sources, edge_destinations], dim=0)
    edge_type = sequence_lts[3][elem_idx][sequence_lts[3][elem_idx] != -1]
    return (x_indices, edge_index, edge_type)


#######

class GRU_RNN(torch.nn.Module):
    def __init__(self, data, grapharea_size, hidden_state_dim, include_senses):
        super(GRU_RNN, self).__init__()
        self.include_senses = include_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]
        self.hidden_state_dim = hidden_state_dim

        # The embeddings matrix for: senses, globals, definitions, examples (the latter 2 may have gradient set to 0)
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_node = Parameter(torch.tensor([0]), requires_grad=False)


        # We update this memory buffer manually, there is no gradient. We set it as a Parameter to DataParallel-ize it
        self.memory_h1 = Parameter(torch.zeros(size=(1, self.hidden_state_dim)), requires_grad=False)
        self.memory_h2 = Parameter(torch.zeros(size=(1, self.hidden_state_dim)), requires_grad=False)

        # 1st layer with GRU
        self.U_z_1 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
        self.W_z_1 = torch.nn.Linear(in_features=self.d, out_features=self.hidden_state_dim, bias=False)
        self.U_r_1 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
        self.W_r_1 = torch.nn.Linear(in_features=self.d, out_features=self.hidden_state_dim, bias=False)

        self.U_1 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
        self.W_1 = torch.nn.Linear(in_features=self.d, out_features=self.hidden_state_dim, bias=False)
        self.dropout = torch.nn.Dropout(p=0.1)

        # 2nd layer with GRU
        self.U_z_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
        self.W_z_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
        self.U_r_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
        self.W_r_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)

        self.U_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)
        self.W_2 = torch.nn.Linear(in_features=self.hidden_state_dim, out_features=self.hidden_state_dim, bias=False)

        # 2nd part of the network as before: 2 linear layers from the RGCN representation to the logits
        self.linear2global = torch.nn.Linear(in_features=self.hidden_state_dim,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)

        if self.include_senses:
            self.linear2sense = torch.nn.Linear(in_features=self.hidden_state_dim,
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
            padded_sequence = padded_sequence.squeeze()
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

            for x_indices in sequence_lts:
                currentword_embedding = self.X.index_select(dim=0, index=x_indices[0])

                z_1 = torch.sigmoid(self.W_z_1(currentword_embedding) + self.U_z_1(self.memory_h1))
                r_1 = torch.sigmoid(self.W_r_1(currentword_embedding) + self.U_r_1(self.memory_h1))
                h_tilde_1 = torch.tanh(self.dropout(self.W_1(currentword_embedding)) + self.U_1(r_1 * self.memory_h1))
                h1 = z_1 * h_tilde_1 + (torch.tensor(1)-z_1) * self.memory_h1

                self.memory_h1.data.copy_(h1.clone()) # store h in memory

                z_2 = torch.sigmoid(self.W_z_2(h1) + self.U_z_2(self.memory_h2))
                r_2 = torch.sigmoid(self.W_r_2(h1) + self.U_r_2(self.memory_h2))
                h_tilde_2 = torch.tanh(self.dropout(self.W_2(h1)) + self.U_2(r_2 * self.memory_h2))
                h2 = z_2 * h_tilde_2 + (torch.tensor(1) - z_2) * self.memory_h2

                self.memory_h2.data.copy_(h2.clone())  # store h in memory

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


#######


class RNN(torch.nn.Module):
    def __init__(self, data, grapharea_size, hidden_state_dim, include_senses):
        super(RNN, self).__init__()
        self.include_senses = include_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.hidden_state_dim =hidden_state_dim
        self.d = data.x.shape[1]

        # The embeddings matrix for: senses, globals, definitions, examples (the latter 2 will have gradient set to 0)
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_node = Parameter(torch.tensor([0]), requires_grad=False)

        # Input vector x(t) is formed by concatenating the vector w representing current word,
        # and the output from neurons in the context layer s at time t-1.

        # We update this memory buffer manually, there is no gradient. We set it as a Parameter to DataParallel-ize it
        self.memory_h1 = Parameter(torch.zeros(size=(1, self.hidden_state_dim)), requires_grad=False)
        self.memory_h2 = Parameter(torch.zeros(size=(1, self.hidden_state_dim//2)), requires_grad=False)

        self.W1 = torch.nn.Linear(in_features=self.d + self.hidden_state_dim, out_features=self.hidden_state_dim,
                                  bias=True)
        self.dropout_1 = torch.nn.Dropout(p=0.1)
        self.W2 = torch.nn.Linear(in_features=3*self.hidden_state_dim//2, out_features=self.hidden_state_dim//2,
                                  bias=True)
        self.dropout_2 = torch.nn.Dropout(p=0.1)


        # 2nd part of the network as before: 2 linear layers from the RGCN representation to the logits
        self.linear2global = torch.nn.Linear(in_features=self.hidden_state_dim//2,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)

        if self.include_senses:
            self.linear2sense = torch.nn.Linear(in_features=self.hidden_state_dim//2,
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

        # The code has been adapted for when batch_size > n_gpu-s
        # self.memory_previous_rgcnconv does not change since we are proceeding "in parallel"
        for padded_sequence in sequences_in_the_batch_ls:
            padded_sequence = padded_sequence.squeeze()
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

            for x_indices in sequence_lts:
                currentword_embedding = self.X.index_select(dim=0, index=x_indices[0])
                input_x_h1 = torch.cat([currentword_embedding, self.memory_h1], dim=1)
                h1 = tfunc.leaky_relu(
                    self.dropout_1(self.W1(input_x_h1)),
                    negative_slope=0.01)
                self.memory_h1.data.copy_(h1.clone()) # store h in memory

                input_h1_h2 = torch.cat([h1, self.memory_h2], dim=1)
                h2 = tfunc.leaky_relu(
                    self.dropout_2(self.W2(input_h1_h2)),
                    negative_slope=0.01)
                self.memory_h2.data.copy_(h2.clone())  # store h in memory

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