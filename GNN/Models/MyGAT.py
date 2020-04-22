import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc

from GNN.Models.Common import unpack_input_tensor
from Utils import DEVICE
from torch.nn.parameter import Parameter


####################
### 1: GRU + GAT ###
####################

class GRU_GAT(torch.nn.Module):
    def __init__(self, data, grapharea_size, num_gat_heads, include_senses):
        super(GRU_GAT, self).__init__()
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


# class GRU_GAT(torch.nn.Module):
#     def __init__(self, data, grapharea_size, senses_attention_heads, include_senses):
#         super(GRU_GAT, self).__init__()
#         self.data = data
#         self.include_senses = include_senses
#         self.last_idx_senses = data.node_types.tolist().index(1)
#         self.last_idx_globals = data.node_types.tolist().index(2)
#         self.N = grapharea_size
#         self.d = data.x.shape[1]
#
#         self.h1_state_dim = 2 * self.d
#         self.h2_state_dim = self.d
#
#         # The embeddings matrix for: senses, globals, definitions, examples
#         self.X = Parameter(data.x.clone().detach(), requires_grad=True)
#         self.select_first_node = Parameter(torch.tensor([0]), requires_grad=False)
#         self.nodestate_zeros = Parameter(torch.zeros(size=(1,self.d)), requires_grad=False)
#
#         # Input signals: current global’s word embedding || global’s node-state (|| sense’s node state)
#         self.concatenated_input_dim = 2 * self.d if not (self.include_senses) else 3 * self.d
#
#         # GAT
#         self.gat_globals = GATConv(in_channels=self.d,
#                                    out_channels=self.d // senses_attention_heads, heads=senses_attention_heads, concat=True,
#                                    negative_slope=0.2, dropout=0, bias=True)
#         if self.include_senses:
#             self.gat_senses = GATConv(in_channels=self.d,
#                                       out_channels=self.d // senses_attention_heads, heads=senses_attention_heads, concat=True,
#                                       negative_slope=0.2, dropout=0, bias=True)
#         # self.gat_out_channels = self.d // (num_attention_heads // 2)
#
#
#         # GRU: we update these memory buffers manually, there is no gradient. Set as a Parameter to DataParallel-ize it
#         self.memory_h1 = Parameter(torch.zeros(size=(1, self.h1_state_dim)), requires_grad=False)
#         self.memory_h2 = Parameter(torch.zeros(size=(1, self.h2_state_dim)), requires_grad=False)
#
#         # GRU: 1st layer
#         self.U_z_1 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h1_state_dim, bias=False)
#         self.W_z_1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim, bias=False)
#         self.U_r_1 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h1_state_dim, bias=False)
#         self.W_r_1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim, bias=False)
#
#         self.U_1 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h1_state_dim, bias=True)
#         self.W_1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim, bias=True)
#         self.dropout = torch.nn.Dropout(p=0.1)
#
#         # GRU: 2nd layer
#         self.U_z_2 = torch.nn.Linear(in_features=self.h2_state_dim, out_features=self.h2_state_dim, bias=False)
#         self.W_z_2 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h2_state_dim, bias=False)
#         self.U_r_2 = torch.nn.Linear(in_features=self.h2_state_dim, out_features=self.h2_state_dim, bias=False)
#         self.W_r_2 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h2_state_dim, bias=False)
#
#         self.U_2 = torch.nn.Linear(in_features=self.h2_state_dim, out_features=self.h2_state_dim, bias=True)
#         self.W_2 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h2_state_dim, bias=True)
#
#
#         # 2nd part of the network: globals' logits and senses' self-attention prediction
#         self.linear2global = torch.nn.Linear(in_features=self.h2_state_dim,
#                                              out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
#
#         if self.include_senses:
#             self.k = 500 # the number of "likely globals". We choose among <=k senses
#             self.likely_globals_embs = Parameter(-1 * torch.ones(size=(self.k, self.d)), requires_grad=False)
#             self.d_qkv = 150 # the dimensionality of queries, keys and values - down from self.d(embeddings)
#             self.mySelfAttention = SelfAttention(dim_input_context=self.concatenated_input_dim, dim_input_elems=self.d,
#                                                  dim_qkv=self.d_qkv, num_multiheads=1) # current number of attention heads=1
#             self.linear2senses = torch.nn.Linear(in_features=self.d_qkv * 1,
#                                                  out_features=self.last_idx_senses, bias=True)
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
#             for (x_indices, edge_index, edge_type), (x_indices_s, edge_index_s, edge_type_s) in sequence_lts:
#                 #t0= time()
#                 # Input signal n.1: the current (global) word
#                 currentword_embedding = self.X.index_select(dim=0, index=x_indices[0])
#
#                 # Input signal n.2: the node-state of the word in the KB-graph, obtained applying the GNN
#                 x = self.X.index_select(dim=0, index=x_indices.squeeze())
#                 x_attention_state = self.gat_globals(x, edge_index)
#                 currentword_node_state = x_attention_state.index_select(dim=0, index=self.select_first_node)
#
#                 # Input signal n.3: the node-state of the current sense; + concatenating the input signals
#                 if self.include_senses:
#                     if x_indices_s.nonzero().shape[0] == 0: # no sense was specified
#                         currentsense_node_state = self.nodestate_zeros
#                     else: # sense was specified
#                         x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
#                         x_attention_state_s = self.gat_senses(x_s, edge_index_s)
#                         currentsense_node_state = x_attention_state_s.index_select(dim=0,index=self.select_first_node)
#                     input_signals = torch.cat([currentword_embedding, currentword_node_state, currentsense_node_state], dim=1)
#                 else:
#                     input_signals = torch.cat([currentword_embedding, currentword_node_state], dim=1)
#
#                 # GRU: Layer 1
#                 z_1 = torch.sigmoid(self.W_z_1(input_signals) + self.U_z_1(self.memory_h1))
#                 r_1 = torch.sigmoid(self.W_r_1(input_signals) + self.U_r_1(self.memory_h1))
#                 h_tilde_1 = torch.tanh(self.dropout(self.W_1(input_signals)) + self.U_1(r_1 * self.memory_h1))
#                 h1 = z_1 * h_tilde_1 + (torch.tensor(1)-z_1) * self.memory_h1
#
#                 self.memory_h1.data.copy_(h1.clone().detach()) # store h in memory
#
#                 # GRU: Layer 2  - globals task
#                 z_2 = torch.sigmoid(self.W_z_2(h1) + self.U_z_2(self.memory_h2))
#                 r_2 = torch.sigmoid(self.W_r_2(h1) + self.U_r_2(self.memory_h2))
#                 h_tilde_2 = torch.tanh(self.dropout(self.W_2(h1)) + self.U_2(r_2 * self.memory_h2))
#                 h2 = z_2 * h_tilde_2 + (torch.tensor(1) - z_2) * self.memory_h2
#
#                 self.memory_h2.data.copy_(h2.clone().detach())  # store h in memory
#
#
#                 # 2nd part of the architecture: predictions
#
#                 # Globals
#                 logits_global = self.linear2global(h2)
#                 sample_predictions_globals = tfunc.log_softmax(logits_global, dim=1)
#                 predictions_globals_ls.append(sample_predictions_globals)
#
#                 # Senses
#                 if self.include_senses:
#                     most_likely_globals = torch.sort(logits_global, descending=True)[1] \
#                                                     [0][0:self.k] + self.last_idx_senses
#
#                     # # for every one of the most likely globals, retrieve the neighbours,
#                     # neighbours_indices_ls = [node_idx for neighbours_subls in list(map(
#                     #     lambda global_node_idx:
#                     #         GA.get_indices_area_toinclude(self.data.edge_index, self.data.edge_type,
#                     #                                       global_node_idx.cpu().item(), area_size=32, max_hops=1)[0],
#                     #     most_likely_globals))
#                     # for node_idx in neighbours_subls]
#
#                     # # and keep only their neighbours that are (in the range of) sense nodes
#                     # likely_senses_indices = torch.tensor(list(filter(
#                     #     lambda node_idx : node_idx in range(0, self.last_idx_senses),
#                     #     neighbours_indices_ls)))[0:self.k]
#
#                     # if torch.cuda.is_available():
#                     #     likely_senses_indices = likely_senses_indices.to("cuda:"+str(torch.cuda.current_device()))
#                     self.likely_globals_embs.data = self.X.index_select(dim=0, index=most_likely_globals)
#
#                     # Self-attention:
#                     selfattention_result = self.mySelfAttention(input_q=input_signals, input_kv=self.likely_globals_embs, k=self.k)
#
#                     # followed by linear layer to the senses' logits
#                     logits_senses = self.linear2senses(selfattention_result)
#                     sample_predictions_senses = tfunc.log_softmax(logits_senses, dim=0)
#                     predictions_senses_ls.append(sample_predictions_senses)
#
#                 else:
#                     predictions_senses_ls.append(torch.tensor(0).to(DEVICE)) # so I don't have to change the interface elsewhere
#
#         return torch.stack(predictions_globals_ls, dim=0).squeeze(), \
#                torch.stack(predictions_senses_ls, dim=0).squeeze()
