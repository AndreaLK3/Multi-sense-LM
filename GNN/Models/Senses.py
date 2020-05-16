import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Graph.Adjacencies as AD
from GNN.Models.Common import unpack_input_tensor
from GNN.Models.Common import SelfAttention
from Utils import DEVICE
from torch.nn.parameter import Parameter




# *****************************

# Choose among the senses of the most likely 5 globals.
# Multiply [the probability distribution over those] per [the distribution over the whole senses' vocabulary].

class SelectK(torch.nn.Module):

    def __init__(self, data, grapharea_matrix, grapharea_size, include_senses_input, predict_senses, batch_size, n_layers=3, n_units=1150):
        super(SelectK, self).__init__()
        self.include_senses_input = include_senses_input
        self.predict_senses = predict_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.grapharea_matrix_lil = grapharea_matrix
        self.N = grapharea_size
        self.d = data.x.shape[1]
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_units = n_units

        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_indices = Parameter(torch.tensor([0,1,2,3,4,5]).to(DEVICE), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.d)), requires_grad=False)

        # Input signals: current global’s word embedding (|| sense’s node state)
        self.concatenated_input_dim = self.d if not (self.include_senses_input) else 2 * self.d

        # This is overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(1, batch_size, n_units)), requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # GRU for globals - standard Language Model
        self.maingru_ls = torch.nn.ModuleList([
            torch.nn.GRU(input_size=self.concatenated_input_dim if i==0 else n_units,
                         hidden_size=n_units, num_layers=1) for i in range(n_layers)])
        # GRU for senses: 1-layer GRU, taking hidden_state_1 from the main GRU (1 layer shared)
        self.gru_senses = torch.nn.GRU(input_size=n_units, hidden_size=n_units, num_layers=1)
        self.k_globals = 5 # how many k most likely globals to use for sense selection

        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=n_units,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)


        self.linear2senses = torch.nn.Linear(in_features=n_units,
                                             out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0

        distributed_batch_size = batchinput_tensor.shape[0]
        if not(distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            new_num_hidden_state_elems = self.n_layers * distributed_batch_size * self.n_units
            self.memory_hn = Parameter(torch.reshape(self.memory_hn.flatten()[0:new_num_hidden_state_elems],
                                           (self.n_layers, distributed_batch_size, self.n_units)), requires_grad=False)
            self.memory_hn_senses = Parameter(
                            torch.reshape(self.memory_hn_senses.flatten()[0:(distributed_batch_size * self.n_units)],
                                          (1, distributed_batch_size, self.n_units)),
                            requires_grad=False)
            self.hidden_state_bsize_adjusted=True
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()
        self.memory_hn_senses.detach_()

        if batchinput_tensor.shape[0] > 1:
            sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
        else:
            sequences_in_the_batch_ls = [batchinput_tensor]

        batch_input_signals_ls = []

        for padded_sequence in sequences_in_the_batch_ls:
            padded_sequence = padded_sequence.squeeze()
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

            sequence_input_signals_ls = []

            for ((x_indices_g, edge_index_g, edge_type_g), (x_indices_s, edge_index_s, edge_type_s)) in sequence_lts:
                # Input signal n.1: the current (global) word
                currentword_embedding = self.X.index_select(dim=0, index=x_indices_g[0])

                # Input signal n.2: the embedding of the current sense; + concatenating the input signals
                if self.include_senses_input:
                    if x_indices_s.nonzero().shape[0] == 0: # no sense was specified
                        currentsense_embedding = self.embedding_zeros
                    else: # sense was specified
                        currentsense_embedding = self.X.index_select(dim=0, index=x_indices_s[0])
                    input_signals = torch.cat([currentword_embedding, currentsense_embedding], dim=1)
                else:
                    input_signals = currentword_embedding

                sequence_input_signals_ls.append(input_signals)
            sequence_input_signals = torch.cat(sequence_input_signals_ls, dim=0).unsqueeze(1)
            batch_input_signals_ls.append(sequence_input_signals)
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=1)

        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        # - h_0 of shape (num_layers * num_directions, batch=1, hidden_size):
        #       tensor containing the initial hidden state for each element in the batch.
        input = batch_input_signals
        gru_out_1 = None
        for i in range(self.n_layers):
            layer_gru = self.maingru_ls[i]
            layer_gru.flatten_parameters()
            main_gru_out, hidden_i = layer_gru(input, self.memory_hn.index_select(dim=0, index=self.select_first_indices[i]))
            self.memory_hn[i].data.copy_(hidden_i.squeeze().clone()) # store h in memory
            input = main_gru_out
            if i==0:
                main_gru_out_layer1=main_gru_out

        main_gru_out = main_gru_out.permute(1,0,2) # going to: (batch_size, seq_len, n_units)
        seq_len = len(sequences_in_the_batch_ls[0][0])
        main_gru_out = main_gru_out.reshape(distributed_batch_size * seq_len, main_gru_out.shape[2])

        # 2nd part of the architecture: predictions
        # globals
        logits_global = self.linear2global(main_gru_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)
        # senses
        # line 1: GRU hidden state + linear.
        self.gru_senses.flatten_parameters()
        senses_gru_out, hidden_s =self.gru_senses(main_gru_out_layer1, self.memory_hn_senses)
        self.memory_hn_senses.data.copy_(hidden_s.clone())  # store h in memory
        senses_gru_out = senses_gru_out.reshape(distributed_batch_size * seq_len, senses_gru_out.shape[2])
        logits_sense = self.linear2senses(senses_gru_out)
        # line 2: select senses of the k most likely globals
        k_globals_indices = logits_global.sort(descending=True).indices[:,0:self.k_globals]
        # k_grapharea_rows = self.grapharea_tensor.tolil()[logits_global.sort(descending=True).indices[:, 0:self.k_globals].flatten()]
        # k_globals_indices = k_globals_indices.reshape(distributed_batch_size * seq_len, k_globals_indices.shape[2])
        neighbouring_nodes_ls = [AD.get_node_data(self.grapharea_matrix_lil, i, self.N, features_mask=(True,False,False))[0]
                                 for i in k_globals_indices.flatten()]
        neighbouring_sense_nodes_ls = list(map(lambda neighbs_t: neighbs_t[neighbs_t < self.last_idx_senses],neighbouring_nodes_ls)) # 160 tensors of variable size, containing the indices of the senses
        
        for senses_t in neighbouring_sense_nodes_ls:
            senses_embeddings = self.X.index_select(dim=0, index=senses_t)


        predictions_senses = tfunc.log_softmax(logits_sense, dim=1)

        return predictions_globals, predictions_senses



# *****************************

# Baseline 2: GRUs only. No shared layers.
class GRU_base2(torch.nn.Module):

    def __init__(self, data, grapharea_size, include_senses_input, predict_senses, batch_size, n_layers=3, n_units=1150):
        super(GRU_base2, self).__init__()
        self.include_senses_input = include_senses_input
        self.predict_senses = predict_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_units = n_units

        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_indices = Parameter(torch.tensor([0,1,2,3,4,5]).to(torch.float32), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.d)), requires_grad=False)

        # Input signals: current global’s word embedding (|| sense’s node state)
        self.concatenated_input_dim = self.d if not (self.include_senses_input) else 2 * self.d

        # This is overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers-1, batch_size, n_units)), requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # GRU for globals - standard Language Model
        self.maingru_ls = torch.nn.ModuleList([
            torch.nn.GRU(input_size=self.concatenated_input_dim if i==0 else n_units,
                         hidden_size=n_units, num_layers=1) for i in range(n_layers)])
        # GRU for senses: 1-layer GRU, taking hidden_state_1 from the main GRU (1 layer shared)
        self.gru_senses = torch.nn.GRU(input_size=self.concatenated_input_dim, hidden_size=n_units, num_layers=n_layers-1)
        if self.include_senses_input:
            self.gat_senses = GATConv(in_channels=self.d, out_channels=int(self.d/4), heads=4)

        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=n_units,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)


        self.linear2senses = torch.nn.Linear(in_features=n_units,
                                             out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0

        distributed_batch_size = batchinput_tensor.shape[0]
        if not(distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            new_num_hidden_state_elems = self.n_layers * distributed_batch_size * self.n_units
            self.memory_hn = Parameter(torch.reshape(self.memory_hn.flatten()[0:new_num_hidden_state_elems],
                                           (self.n_layers, distributed_batch_size, self.n_units)), requires_grad=False)
            self.memory_hn_senses = Parameter(
                            torch.reshape(self.memory_hn_senses.flatten()[0:((self.n_layers-1) *distributed_batch_size * self.n_units)],
                                          (self.n_layers-1, distributed_batch_size, self.n_units)),
                            requires_grad=False)
            self.hidden_state_bsize_adjusted=True
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()
        self.memory_hn_senses.detach_()

        if batchinput_tensor.shape[0] > 1:
            sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
        else:
            sequences_in_the_batch_ls = [batchinput_tensor]

        batch_input_signals_ls = []

        for padded_sequence in sequences_in_the_batch_ls:
            padded_sequence = padded_sequence.squeeze()
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

            sequence_input_signals_ls = []

            for ((x_indices_g, edge_index_g, edge_type_g), (x_indices_s, edge_index_s, edge_type_s)) in sequence_lts:
                # Input signal n.1: the current (global) word
                currentword_embedding = self.X.index_select(dim=0, index=x_indices_g[0])

                # Input signal n.2: the node-state of the current sense; + concatenating the input signals
                if self.include_senses_input:
                    if len(x_indices_s[x_indices_s != 0] == 0):  # no sense was specified
                        currentsense_node_state = self.embedding_zeros
                    else:  # sense was specified
                        x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
                        sense_attention_state = self.gat_senses(x_s, edge_index_s)
                        currentsense_node_state = sense_attention_state.index_select(dim=0,
                                                                                     index=self.select_first_indices[0].to(torch.int64))
                    input_signals = torch.cat([currentword_embedding, currentsense_node_state], dim=1)
                else:
                    input_signals = currentword_embedding
                sequence_input_signals_ls.append(input_signals)


            sequence_input_signals = torch.cat(sequence_input_signals_ls, dim=0).unsqueeze(1)
            batch_input_signals_ls.append(sequence_input_signals)
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=1)

        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        # - h_0 of shape (num_layers * num_directions, batch=1, hidden_size):
        #       tensor containing the initial hidden state for each element in the batch.
        input = batch_input_signals
        for i in range(self.n_layers):
            layer_gru = self.maingru_ls[i]
            layer_gru.flatten_parameters()
            main_gru_out, hidden_i = layer_gru(input, self.memory_hn.index_select(dim=0, index=self.select_first_indices[i].to(torch.int64)))
            self.memory_hn[i].data.copy_(hidden_i.squeeze().clone()) # store h in memory
            input = main_gru_out

        main_gru_out = main_gru_out.permute(1,0,2) # going to: (batch_size, seq_len, n_units)
        seq_len = len(sequences_in_the_batch_ls[0][0])
        main_gru_out = main_gru_out.reshape(distributed_batch_size * seq_len, main_gru_out.shape[2])

        # 2nd part of the architecture: predictions
        # globals
        logits_global = self.linear2global(main_gru_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)
        # senses
        # line 1: GRU hidden state + linear.
        self.gru_senses.flatten_parameters()
        senses_gru_out, hidden_s =self.gru_senses(batch_input_signals, self.memory_hn_senses)
        self.memory_hn_senses.data.copy_(hidden_s.clone())  # store h in memory
        senses_gru_out = senses_gru_out.reshape(distributed_batch_size * seq_len, senses_gru_out.shape[2])
        logits_sense = self.linear2senses(senses_gru_out)
        # line 2: select senses of the k most likely globals
        # TO BE IMPLEMENTED in another version of the model

        predictions_senses = tfunc.log_softmax(logits_sense, dim=1)


        return predictions_globals, predictions_senses





# ******************************

# *****************************

# Baseline 1: GRUs only. 1 shared layer (the 1st) betwee the 2 GRUs

class GRU_base(torch.nn.Module):

    def __init__(self, data, grapharea_size, include_senses_input, predict_senses, batch_size, n_layers=3, n_units=1150):
        super(GRU_base, self).__init__()
        self.include_senses_input = include_senses_input
        self.predict_senses = predict_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_units = n_units

        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_indices = Parameter(torch.tensor([0,1,2,3,4,5]).to(torch.float32), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.d)), requires_grad=False)

        # Input signals: current global’s word embedding (|| sense’s node state)
        self.concatenated_input_dim = self.d if not (self.include_senses_input) else 2 * self.d

        # This is overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(1, batch_size, n_units)), requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # GRU for globals - standard Language Model
        self.maingru_ls = torch.nn.ModuleList([
            torch.nn.GRU(input_size=self.concatenated_input_dim if i==0 else n_units,
                         hidden_size=n_units, num_layers=1) for i in range(n_layers)])
        # GRU for senses: 1-layer GRU, taking hidden_state_1 from the main GRU (1 layer shared)
        self.gru_senses = torch.nn.GRU(input_size=n_units, hidden_size=n_units, num_layers=1)

        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=n_units,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)

        self.linear2senses = torch.nn.Linear(in_features=n_units,
                                             out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0

        distributed_batch_size = batchinput_tensor.shape[0]
        if not(distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            new_num_hidden_state_elems = self.n_layers * distributed_batch_size * self.n_units
            self.memory_hn = Parameter(torch.reshape(self.memory_hn.flatten()[0:new_num_hidden_state_elems],
                                           (self.n_layers, distributed_batch_size, self.n_units)), requires_grad=False)
            self.memory_hn_senses = Parameter(
                            torch.reshape(self.memory_hn_senses.flatten()[0:(distributed_batch_size * self.n_units)],
                                          (1, distributed_batch_size, self.n_units)),
                            requires_grad=False)
            self.hidden_state_bsize_adjusted=True
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()
        self.memory_hn_senses.detach_()

        if batchinput_tensor.shape[0] > 1:
            sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
        else:
            sequences_in_the_batch_ls = [batchinput_tensor]

        batch_input_signals_ls = []

        for padded_sequence in sequences_in_the_batch_ls:
            padded_sequence = padded_sequence.squeeze()
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

            sequence_input_signals_ls = []

            for ((x_indices_g, edge_index_g, edge_type_g), (x_indices_s, edge_index_s, edge_type_s)) in sequence_lts:
                # Input signal n.1: the current (global) word
                currentword_embedding = self.X.index_select(dim=0, index=x_indices_g[0])

                # Input signal n.2: the embedding of the current sense; + concatenating the input signals
                if self.include_senses_input:
                    if x_indices_s.nonzero().shape[0] == 0: # no sense was specified
                        currentsense_embedding = self.embedding_zeros
                    else: # sense was specified
                        currentsense_embedding = self.X.index_select(dim=0, index=x_indices_s[0])
                    input_signals = torch.cat([currentword_embedding, currentsense_embedding], dim=1)
                else:
                    input_signals = currentword_embedding

                sequence_input_signals_ls.append(input_signals)
            sequence_input_signals = torch.cat(sequence_input_signals_ls, dim=0).unsqueeze(1)
            batch_input_signals_ls.append(sequence_input_signals)
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=1)

        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        # - h_0 of shape (num_layers * num_directions, batch=1, hidden_size):
        #       tensor containing the initial hidden state for each element in the batch.
        input = batch_input_signals
        gru_out_1 = None
        for i in range(self.n_layers):
            layer_gru = self.maingru_ls[i]
            layer_gru.flatten_parameters()
            main_gru_out, hidden_i = layer_gru(input, self.memory_hn.index_select(dim=0, index=self.select_first_indices[i]))
            self.memory_hn[i].data.copy_(hidden_i.squeeze().clone()) # store h in memory
            input = main_gru_out
            if i==0:
                main_gru_out_layer1=main_gru_out

        main_gru_out = main_gru_out.permute(1,0,2) # going to: (batch_size, seq_len, n_units)
        seq_len = len(sequences_in_the_batch_ls[0][0])
        main_gru_out = main_gru_out.reshape(distributed_batch_size * seq_len, main_gru_out.shape[2])

        # 2nd part of the architecture: predictions
        # globals
        logits_global = self.linear2global(main_gru_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)
        # senses
        # line 1: GRU hidden state + linear.
        self.gru_senses.flatten_parameters()
        senses_gru_out, hidden_s =self.gru_senses(main_gru_out_layer1, self.memory_hn_senses)
        self.memory_hn_senses.data.copy_(hidden_s.clone())  # store h in memory
        senses_gru_out = senses_gru_out.reshape(distributed_batch_size * seq_len, senses_gru_out.shape[2])
        logits_sense = self.linear2senses(senses_gru_out)
        # line 2: select senses of the k most likely globals
        # TO BE IMPLEMENTED in another version of the model

        predictions_senses = tfunc.log_softmax(logits_sense, dim=1)


        return predictions_globals, predictions_senses





# ******************************

class SelfAttK(torch.nn.Module):
    def __init__(self, data, grapharea_size, num_gat_heads, include_senses, num_senses_attheads=1):
        super(SelfAttK, self).__init__()
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
            self.k = 10 # the number of "likely globals".
            self.d_qkv = self.d # the dimensionality of queries, keys and values - down from self.d(embeddings)
            self.mySelfAttention = SelfAttention(dim_input_context=self.concatenated_input_dim, dim_input_elems=self.d,
                                                 dim_qkv=self.d_qkv, num_multiheads=num_senses_attheads)
            self.linear2senses = torch.nn.Linear(in_features=self.d_qkv * num_senses_attheads,
                                                 out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0

        predictions_globals_ls = []
        predictions_senses_ls = []
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_h1.detach_()
        self.memory_h2.detach_()
        #self.k_globals_embds.detach_()

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
                    #k_logits = logits_global_sorted[0].squeeze()[0:self.k]
                    k_globals = logits_global_sorted[1].squeeze()[0:self.k]
                    k_globals_indicesX = k_globals + self.last_idx_senses

                    k_globals_embds=self.X.index_select(dim=0, index=k_globals_indicesX)
                    att_result = self.mySelfAttention(input_q=input_signals, input_kv=k_globals_embds, k=self.k)
                    logits_sense = self.linear2senses(att_result)

                    sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
                    predictions_senses_ls.append(sample_predictions_senses)
                else:
                    predictions_senses_ls.append(torch.tensor(0).to(DEVICE)) # so I don't have to change the interface elsewhere

            #Utils.log_chronometer([t0,t1,t2,t3,t4,t5, t6, t7, t8])
        return torch.stack(predictions_globals_ls, dim=0).squeeze(), \
               torch.stack(predictions_senses_ls, dim=0).squeeze()



# ***********************************************
# class ProjectK(torch.nn.Module):
#     def __init__(self, data, grapharea_size, num_gat_heads, include_senses):
#         super(ProjectK, self).__init__()
#         self.include_senses = include_senses
#         self.last_idx_senses = data.node_types.tolist().index(1)
#         self.last_idx_globals = data.node_types.tolist().index(2)
#         self.N = grapharea_size
#         self.d = data.x.shape[1]
#
#         self.h1_state_dim = 2 * self.d if self.include_senses else self.d
#         self.h2_state_dim = self.d
#
#         # The embeddings matrix for: senses, globals, definitions, examples (the latter 2 may have gradient set to 0)
#         self.X = Parameter(data.x.clone().detach(), requires_grad=True)
#         self.select_first_node = Parameter(torch.tensor([0]), requires_grad=False)
#         self.embedding_zeros = Parameter(torch.zeros(size=(1, self.d)), requires_grad=False)
#
#         # GAT
#         self.gat_globals = GATConv(in_channels=self.d,
#                                    out_channels=self.d // num_gat_heads, heads=num_gat_heads, concat=True,
#                                    negative_slope=0.2, dropout=0, bias=True)
#         if self.include_senses:
#             self.gat_senses = GATConv(in_channels=self.d,
#                                       out_channels=self.d // num_gat_heads, heads=num_gat_heads, concat=True,
#                                       negative_slope=0.2, dropout=0, bias=True)
#
#         # Input signals: current global’s word embedding || global’s node-state (|| sense’s node state)
#         self.concatenated_input_dim = 2*self.d if not (self.include_senses) else 3 * self.d
#
#         # GRU: we update these memory buffers manually, there is no gradient. Set as a Parameter to DataParallel-ize it
#         self.memory_h1 = Parameter(torch.zeros(size=(1, self.h1_state_dim)), requires_grad=False)
#         self.memory_h2 = Parameter(torch.zeros(size=(1, self.h2_state_dim)), requires_grad=False)
#
#         # GRU: 1st layer
#         self.U_z_1 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h1_state_dim, bias=False)
#         self.W_z_1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim,
#                                      bias=False)
#         self.U_r_1 = torch.nn.Linear(in_features=self.h1_state_dim, out_features=self.h1_state_dim, bias=False)
#         self.W_r_1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=self.h1_state_dim,
#                                      bias=False)
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
#         # 2nd part of the network as before: 2 linear layers to the logits
#         self.linear2global = torch.nn.Linear(in_features=self.h2_state_dim,
#                                              out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
#
#         if self.include_senses:
#             self.k = 20  # the number of "most likely globals".
#             self.k_globals_embds =  Parameter(torch.zeros(size=(self.k, self.d)), requires_grad=False)
#             self.dp = 150
#             self.P = torch.nn.Linear(in_features=self.d, out_features=self.dp, bias=False)
#             self.projs2senselogits = torch.nn.Linear(in_features=self.k*(self.dp+1), out_features=self.last_idx_senses, bias=True)
#
#
#     def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
#
#         predictions_globals_ls = []
#         predictions_senses_ls = []
#         # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
#         self.memory_h1.detach_()
#         self.memory_h2.detach_()
#         self.k_globals_embds.detach_()
#
#         if batchinput_tensor.shape[0] > 1:
#             sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
#         else:
#             sequences_in_the_batch_ls = [batchinput_tensor]
#
#         for padded_sequence in sequences_in_the_batch_ls:
#             padded_sequence = padded_sequence.squeeze()
#             padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
#             sequence_lts = [unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]
#
#             for ((x_indices, edge_index, edge_type), (x_indices_s, edge_index_s, edge_type_s)) in sequence_lts:
#                 # Input signal n.1: the embedding of the current (global) word
#                 currentword_embedding = self.X.index_select(dim=0, index=x_indices[0])
#
#                 # Input signal n.2: the node-state of the current global word
#                 x = self.X.index_select(dim=0, index=x_indices.squeeze())
#                 x_attention_state = self.gat_globals(x, edge_index)
#                 currentglobal_node_state = x_attention_state.index_select(dim=0, index=self.select_first_node)
#
#                 # Input signal n.3: the node-state of the current sense; + concatenating the input signals
#                 if self.include_senses:
#                     if x_indices_s.nonzero().shape[0] == 0: # no sense was specified
#                         currentsense_node_state = self.embedding_zeros
#                     else: # sense was specified
#                         x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
#                         sense_attention_state = self.gat_senses(x_s, edge_index_s)
#                         currentsense_node_state = sense_attention_state.index_select(dim=0, index=self.select_first_node)
#                     input_signals = torch.cat([currentword_embedding, currentglobal_node_state, currentsense_node_state], dim=1)
#                 else:
#                     input_signals = torch.cat([currentword_embedding, currentglobal_node_state], dim=1)
#
#                 # GRU: Layer 1
#                 z_1 = torch.sigmoid(self.W_z_1(input_signals) + self.U_z_1(self.memory_h1))
#                 r_1 = torch.sigmoid(self.W_r_1(input_signals) + self.U_r_1(self.memory_h1))
#                 h_tilde_1 = torch.tanh(self.dropout(self.W_1(input_signals)) + self.U_1(r_1 * self.memory_h1))
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
#                 # 2nd partof the architecture: predictions
#                 # Globals
#                 logits_global = self.linear2global(h2)  # shape=torch.Size([5])
#                 sample_predictions_globals = tfunc.log_softmax(logits_global, dim=1)
#                 predictions_globals_ls.append(sample_predictions_globals)
#                 # Senses
#                 if self.include_senses:
#                     logits_global_sorted = torch.sort(logits_global, descending=True)
#                     k_logits = logits_global_sorted[0].squeeze()[0:self.k]
#                     k_globals = logits_global_sorted[1].squeeze()[0:self.k]
#                     k_globals_indicesX = k_globals + self.last_idx_senses
#
#                     self.k_globals_embds.data.copy_(self.X.index_select(dim=0, index=k_globals_indicesX).clone())
#                     projs = self.P(self.k_globals_embds)
#                     p_l = torch.cat([projs, k_logits.unsqueeze(dim=1)], dim=1)
#                     logits_sense = self.projs2senselogits(p_l.flatten())
#
#                     sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
#                     predictions_senses_ls.append(sample_predictions_senses)
#                 else:
#                     predictions_senses_ls.append(torch.tensor(0).to(DEVICE)) # so I don't have to change the interface elsewhere
#
#             #Utils.log_chronometer([t0,t1,t2,t3,t4,t5, t6, t7, t8])
#         return torch.stack(predictions_globals_ls, dim=0).squeeze(), \
#                torch.stack(predictions_senses_ls, dim=0).squeeze()