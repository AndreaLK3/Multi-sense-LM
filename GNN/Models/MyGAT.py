import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
from math import sqrt
import GNN.Models.Common as Common
from Utils import DEVICE
from torch.nn.parameter import Parameter
from Utils import log_chronometer
from time import time
import numpy as np

####################
### 1: GRU + GAT ###
####################

class GRU_GAT(torch.nn.Module):

    def __init__(self, data, grapharea_size, num_gat_heads, include_senses, batch_size, n_layers, n_units):
        super(GRU_GAT, self).__init__()
        self.include_senses = include_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]
        self.batch_size = batch_size

        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_node = Parameter(torch.tensor([0]).to(DEVICE), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.d)), requires_grad=False)

        # GAT
        n_units_in_one_att_head = int(self.d *sqrt(num_gat_heads) // num_gat_heads)
        self.gat_globals = GATConv(in_channels=self.d,
                                   out_channels=n_units_in_one_att_head, heads=num_gat_heads, concat=True,
                                   negative_slope=0.2, dropout=0, bias=True)
        if self.include_senses:
            self.gat_senses = GATConv(in_channels=self.d,
                                      out_channels=n_units_in_one_att_head, heads=num_gat_heads, concat=True,
                                      negative_slope=0.2, dropout=0, bias=True)

        # Input signals: current global’s word embedding || global’s node-state (|| sense’s node state)
        self.concatenated_input_dim = self.d + (n_units_in_one_att_head * num_gat_heads) if not (self.include_senses) \
                                 else self.d + 2*(n_units_in_one_att_head * num_gat_heads)

        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_units)), requires_grad=False)

        self.gru = torch.nn.GRU(input_size=self.concatenated_input_dim, hidden_size=n_units, num_layers=n_layers)
        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=n_units,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)

        if self.include_senses:
            self.linear2sense = torch.nn.Linear(in_features=n_units,
                                                out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0

        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()

        t0 = time()
        batchinput_ndarray_0 = batchinput_tensor.cpu().numpy()
        batchinput_ndarray = np.apply_along_axis(func1d= lambda row_in_batch : np.apply_along_axis(
            func1d= lambda sample_tensor: Common.unpack_input_tensor_numpy(sample_tensor, self.N), arr=row_in_batch, axis=0),
        axis=2, arr=batchinput_ndarray_0)
        t1=time()

        # if batchinput_tensor.shape[0] > 1:
        #     sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
        # else:
        #     sequences_in_the_batch_ls = [batchinput_tensor]
        #
        # batch_input_signals_ls = []
        #
        # for padded_sequence in sequences_in_the_batch_ls:
        #     t0 = time()
        #     padded_sequence = padded_sequence.squeeze()
        #     padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
        #     sequence_lts = [Common.unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]
        #
        #     sequence_input_signals_ls = []
        #     t1 = time()
        batch_input_signals_ls = []
        for part_of_batch in batchinput_ndarray:
            sequence_input_signals_ls = []
            for sequence_lts in part_of_batch:
                for ((x_indices_g, edge_index_g, edge_type_g), (x_indices_s, edge_index_s, edge_type_s)) in sequence_lts:
                    # Input signal n.1: the embedding of the current (global) word
                    currentword_embedding = self.X.index_select(dim=0, index=x_indices_g[0])

                    # Input signal n.2: the node-state of the current global word
                    x = self.X.index_select(dim=0, index=x_indices_g.squeeze())
                    # experiment: edge_index as sparse tensor, to accumulate it into batches
                    # edge_index_g_reversed = torch.stack([edge_index_g[1], edge_index_g[0]], dim=0)
                    # edge_index_g_sparse = torch.sparse_coo_tensor(indices = edge_index_g_reversed, values=torch.ones(edge_index_g.shape[1]))
                    x_attention_state = self.gat_globals(x, edge_index_g)
                    currentglobal_node_state = x_attention_state.index_select(dim=0, index=self.select_first_node)

                    # Input signal n.3: the node-state of the current sense; + concatenating the input signals
                    if self.include_senses:
                        if x_indices_s.nonzero().shape[0] == 0:  # no sense was specified
                            currentsense_node_state = self.embedding_zeros
                        else:  # sense was specified
                            x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
                            sense_attention_state = self.gat_senses(x_s, edge_index_s)
                            currentsense_node_state = sense_attention_state.index_select(dim=0,
                                                                                         index=self.select_first_node)
                        input_signals = torch.cat([currentword_embedding, currentglobal_node_state, currentsense_node_state], dim=1)
                    else:
                        input_signals = torch.cat([currentword_embedding, currentglobal_node_state], dim=1)

                    sequence_input_signals_ls.append(input_signals)
            t2 = time()
            sequence_input_signals = torch.cat(sequence_input_signals_ls, dim=0).unsqueeze(1)
            batch_input_signals_ls.append(sequence_input_signals)
            t3 = time()
            log_chronometer([t0, t1, t2, t3])
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=1)


        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        # - h_0 of shape (num_layers * num_directions, batch=1, hidden_size):
        #       tensor containing the initial hidden state for each element in the batch.
        self.gru.flatten_parameters()
        gru_out, hidden_n = self.gru(batch_input_signals, self.memory_hn)
        self.memory_hn.data.copy_(hidden_n.clone()) # store h in memory
        gru_out = gru_out.permute(1,0,2) # going to: (batch_size, seq_len, n_units)
        seq_len = len(sequences_in_the_batch_ls[0][0])
        gru_out = gru_out.reshape(self.batch_size * seq_len, gru_out.shape[2])

        # 2nd part of the architecture: predictions
        logits_global = self.linear2global(gru_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)
        if self.include_senses:
            logits_sense = self.linear2sense(gru_out)
            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0]*self.batch_size * seq_len).to(DEVICE) # so I don't have to change the interface elsewhere

        return predictions_globals, predictions_senses



#######################
### 2: My GRU + GAT ###
#######################

class MyGRU_GAT(torch.nn.Module):
    def __init__(self, data, grapharea_size, num_gat_heads, include_senses):
        super(MyGRU_GAT, self).__init__()
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
        self.dropout = torch.nn.Dropout(p=0.01)

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
            sequence_lts = [Common.unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

            for ((x_indices, edge_index, edge_type), (x_indices_s, edge_index_s, edge_type_s)) in sequence_lts:
                # Input signal n.1: the embedding of the current (global) word
                currentword_embedding = self.X.index_select(dim=0, index=x_indices[0])

                # Input signal n.2: the node-state of the current global word
                x = self.X.index_select(dim=0, index=x_indices.squeeze())
                x_attention_state = self.gat_globals(x, edge_index)
                currentglobal_node_state = x_attention_state.index_select(dim=0, index=self.select_first_node)

                # Input signal n.3: the node-state of the current sense; + concatenating the input signals
                if self.include_senses:
                    if x_indices_s.nonzero().shape[0] == 0:  # no sense was specified
                        currentsense_node_state = self.embedding_zeros
                    else:  # sense was specified
                        x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
                        sense_attention_state = self.gat_senses(x_s, edge_index_s)
                        currentsense_node_state = sense_attention_state.index_select(dim=0,
                                                                                     index=self.select_first_node)
                    input_signals = torch.cat(
                        [currentword_embedding, currentglobal_node_state, currentsense_node_state], dim=1)
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


########################
### 3: WD-LSTM + GAT ###
########################

class WD_LSTM_GAT(torch.nn.Module):
    def __init__(self, data, grapharea_size, num_gat_heads, include_senses, batch_size, n_layers, n_units):
        super(WD_LSTM_GAT, self).__init__()
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.include_senses = include_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]

        # The embeddings matrix for: senses, globals, definitions, examples (the latter 2 may have gradient set to 0)
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_node = Parameter(torch.tensor([0]).to(DEVICE), requires_grad=False)
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
        self.concatenated_input_dim = 2 * self.d if not (self.include_senses) else 3 * self.d

        # Memories for the hidden and cell states of the LSTM
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_units)), requires_grad=False)
        self.memory_cn = Parameter(torch.zeros(size=(n_layers, batch_size, n_units)), requires_grad=False) # self.d if i==0 else

        #self.wd_lstm = WeightDropLSTM(input_size=self.concatenated_input_dim, num_layers=n_layers, hidden_size=n_units)
        # we must use manual WeightDrop on LSTM cells, WeightDropLSTM is incompatible with PyTorch 1.4.0
        self.lstm = Common.weight_drop(module=torch.nn.LSTM(input_size=self.concatenated_input_dim,
                                                            hidden_size=n_units, num_layers=n_layers),
                                  weights_names_ls=['weight_hh_l'+str(i) for i in range(n_layers)],
                                  dropout_p=0.3)
        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=n_units,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)

        if self.include_senses:
            self.linear2sense = torch.nn.Linear(in_features=n_units,
                                                out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0

        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()
        self.memory_cn.detach_()

        if batchinput_tensor.shape[0] > 1:
            sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
        else:
            sequences_in_the_batch_ls = [batchinput_tensor]

        batch_input_signals_ls = []

        for padded_sequence in sequences_in_the_batch_ls:
            padded_sequence = padded_sequence.squeeze()
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [Common.unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

            sequence_input_signals_ls = []

            for ((x_indices_g, edge_index_g, edge_type_g), (x_indices_s, edge_index_s, edge_type_s)) in sequence_lts:
                # Input signal n.1: the embedding of the current (global) word
                currentword_embedding = self.X.index_select(dim=0, index=x_indices_g[0])

                # Input signal n.2: the node-state of the current global word
                x = self.X.index_select(dim=0, index=x_indices_g.squeeze())
                x_attention_state = self.gat_globals(x, edge_index_g)
                currentglobal_node_state = x_attention_state.index_select(dim=0, index=self.select_first_node)

                # Input signal n.3: the node-state of the current sense; + concatenating the input signals
                if self.include_senses:
                    if x_indices_s.nonzero().shape[0] == 0:  # no sense was specified
                        currentsense_node_state = self.embedding_zeros
                    else:  # sense was specified
                        x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
                        sense_attention_state = self.gat_senses(x_s, edge_index_s)
                        currentsense_node_state = sense_attention_state.index_select(dim=0,
                                                                                     index=self.select_first_node)
                    input_signals = torch.cat(
                        [currentword_embedding, currentglobal_node_state, currentsense_node_state], dim=1)
                else:
                    input_signals = torch.cat([currentword_embedding, currentglobal_node_state], dim=1)

                sequence_input_signals_ls.append(input_signals)
            sequence_input_signals = torch.cat(sequence_input_signals_ls, dim=0).unsqueeze(1)
            batch_input_signals_ls.append(sequence_input_signals)
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=1)


        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        # - h_0/c_0 of shape (num_layers * num_directions, batch=1, hidden_size):
        #       tensor containing the initial hidden state/cell state for each element in the batch.
        self.lstm.flatten_parameters()
        lstm_out, (hidden_n, cells_n) = self.lstm(batch_input_signals, (self.memory_hn, self.memory_cn))
        self.memory_hn.data.copy_(hidden_n.clone()) # store h in memory
        self.memory_cn.data.copy_(cells_n.clone())
        lstm_out = lstm_out.permute(1,0,2) # going to: (batch_size, seq_len, n_units)
        seq_len = len(sequences_in_the_batch_ls[0][0])
        lstm_out = lstm_out.reshape(self.batch_size * seq_len, lstm_out.shape[2])

        # 2nd part of the architecture: predictions
        logits_global = self.linear2global(lstm_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)
        if self.include_senses:
            logits_sense = self.linear2sense(lstm_out)
            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0]*self.batch_size * seq_len).to(DEVICE) # so I don't have to change the interface elsewhere

        return predictions_globals, predictions_senses


# ************************************************************