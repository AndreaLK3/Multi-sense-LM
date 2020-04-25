import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Graph.GraphArea as GA
from GNN.Models.Common import unpack_input_tensor
from Utils import DEVICE
from torch.nn.parameter import Parameter


class ProjectK(torch.nn.Module):
    def __init__(self, data, grapharea_size, num_gat_heads, include_senses):
        super(ProjectK, self).__init__()
        self.include_senses = include_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]

        self.h1_state_dim = 2 * self.d if self.include_senses else self.d
        self.h2_state_dim = self.d

        # The embeddings matrix for: senses, globals, definitions, examples (the latter 2 may have gradient set to 0)
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_node = Parameter(torch.tensor([0]), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.d)), requires_grad=False)

        # GAT
        self.gat_globals = GATConv(in_channels=self.d,
                                   out_channels=self.d // num_gat_heads, heads=num_gat_heads, concat=True,
                                   negative_slope=0.2, dropout=0, bias=True)
        if self.include_senses:
            self.gat_senses = GATConv(in_channels=self.d,
                                      out_channels=self.d // num_gat_heads, heads=num_gat_heads, concat=True,
                                      negative_slope=0.2, dropout=0, bias=True)

        # Input signals: current global’s word embedding || global’s node-state (|| sense’s node state)
        self.concatenated_input_dim = 2*self.d if not (self.include_senses) else 3 * self.d

        # GRU: we update these memory buffers manually, there is no gradient. Set as a Parameter to DataParallel-ize it
        self.memory_h1 = Parameter(torch.zeros(size=(1, self.h1_state_dim)), requires_grad=False)
        self.memory_h2 = Parameter(torch.zeros(size=(1, self.h2_state_dim)), requires_grad=False)

        # GRU: 1st layer
        self.U_z_1 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h1_state_dim, bias=False)
        self.W_z_1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim,
                                     bias=False)
        self.U_r_1 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h1_state_dim, bias=False)
        self.W_r_1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim,
                                     bias=False)

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
            self.k = 20  # the number of "most likely globals".
            self.k_globals_embds =  Parameter(torch.zeros(size=(self.k, self.d)), requires_grad=False)
            self.dp = 150
            self.P = torch.nn.Linear(in_features=self.d, out_features=self.dp, bias=False)
            self.projs2senselogits = torch.nn.Linear(in_features=self.k*(self.dp+1), out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0

        predictions_globals_ls = []
        predictions_senses_ls = []
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_h1.detach_()
        self.memory_h2.detach_()
        self.k_globals_embds.detach_()

        if batchinput_tensor.shape[0] > 1:
            sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
        else:
            sequences_in_the_batch_ls = [batchinput_tensor]

        for padded_sequence in sequences_in_the_batch_ls:
            padded_sequence = padded_sequence.squeeze()
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

            for ((x_indices, edge_index, edge_type), (x_indices_s, edge_index_s, edge_type_s)) in sequence_lts:
                # Input signal n.1: the embedding of the current (global) word
                currentword_embedding = self.X.index_select(dim=0, index=x_indices[0])

                # Input signal n.2: the node-state of the current global word
                x = self.X.index_select(dim=0, index=x_indices.squeeze())
                x_attention_state = self.gat_globals(x, edge_index)
                currentglobal_node_state = x_attention_state.index_select(dim=0, index=self.select_first_node)

                # Input signal n.3: the node-state of the current sense; + concatenating the input signals
                if self.include_senses:
                    if x_indices_s.nonzero().shape[0] == 0: # no sense was specified
                        currentsense_node_state = self.embedding_zeros
                    else: # sense was specified
                        x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
                        sense_attention_state = self.gat_senses(x_s, edge_index_s)
                        currentsense_node_state = sense_attention_state.index_select(dim=0, index=self.select_first_node)
                    input_signals = torch.cat([currentword_embedding, currentglobal_node_state, currentsense_node_state], dim=1)
                else:
                    input_signals = torch.cat([currentword_embedding, currentglobal_node_state], dim=1)

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

                # 2nd partof the architecture: predictions
                # Globals
                logits_global = self.linear2global(h2)  # shape=torch.Size([5])
                sample_predictions_globals = tfunc.log_softmax(logits_global, dim=1)
                predictions_globals_ls.append(sample_predictions_globals)
                # Senses
                if self.include_senses:
                    logits_global_sorted = torch.sort(logits_global, descending=True)
                    k_logits = logits_global_sorted[0].squeeze()[0:self.k]
                    k_globals = logits_global_sorted[1].squeeze()[0:self.k]
                    k_globals_indicesX = k_globals + self.last_idx_senses

                    self.k_globals_embds.data.copy_(self.X.index_select(dim=0, index=k_globals_indicesX).clone())
                    projs = self.P(self.k_globals_embds)
                    p_l = torch.cat([projs, k_logits.unsqueeze(dim=1)], dim=1)
                    logits_sense = self.projs2senselogits(p_l.flatten())

                    sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
                    predictions_senses_ls.append(sample_predictions_senses)
                else:
                    predictions_senses_ls.append(torch.tensor(0).to(DEVICE)) # so I don't have to change the interface elsewhere

            #Utils.log_chronometer([t0,t1,t2,t3,t4,t5, t6, t7, t8])
        return torch.stack(predictions_globals_ls, dim=0).squeeze(), \
               torch.stack(predictions_senses_ls, dim=0).squeeze()