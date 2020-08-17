import torch
from torch_geometric.nn import GATConv
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
import torch.nn.functional as tfunc
from GNN.Models.Common import unpack_input_tensor, init_model_parameters, lemmatize_node
from GNN.Models.Steps_RNN import rnn_loop
from torch.nn.parameter import Parameter
import logging
import nltk
from PrepareKBInput.LemmatizeNyms import lemmatize_term






# ****** The model (LSTM / GRU) ******


class RNN(torch.nn.Module):

    def __init__(self, model_type, data, grapharea_size, grapharea_matrix, vocabulary_df, include_globalnode_input, include_sensenode_input, predict_senses,
                 batch_size, n_layers, n_hid_units, dropout_p):
        super(RNN, self).__init__()
        self.model_type = model_type  # can be "LSTM" or "GRU"
        init_model_parameters(self, data, grapharea_size, grapharea_matrix, vocabulary_df,
                              include_globalnode_input, include_sensenode_input, predict_senses,
                              batch_size, n_layers, n_hid_units, dropout_p)

        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_indices = Parameter(torch.tensor(list(range(n_hid_units))).to(torch.float32), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        # Input signals: current global’s word embedding (|| global's node state (|| sense’s node state) )
        self.concatenated_input_dim = self.dim_embs * (1 + int(include_globalnode_input) + int(include_sensenode_input))

        # This is overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_cn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))), requires_grad=False)
        self.memory_cn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))), requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # RNN for globals - standard Language Model
        self.main_rnn_ls = torch.nn.ModuleList(
            [getattr(torch.nn, self.model_type)(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                              hidden_size=512 if i == n_layers - 1 else n_hid_units, num_layers=1) for i in range(n_layers)]) # 400
        # GAT for the node-states from the dictionary graph
        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=self.dim_embs, out_channels=int(self.dim_embs / 4), heads=4)#, node_dim=1)
            # lemmatize_term('init', self.lemmatizer)# to establish LazyCorpusLoader and prevent a multi-thread crash
        if self.include_sensenode_input:
            self.gat_senses = GATConv(in_channels=self.dim_embs, out_channels=int(self.dim_embs / 4), heads=4)
        # RNN for senses
        if predict_senses:
            self.senses_rnn_ls = torch.nn.ModuleList(
            [getattr(torch.nn, self.model_type)(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                              hidden_size=512 if i == n_layers - 1 else n_hid_units, num_layers=1) for i in range(n_layers)]) # 400

        # 2nd part of the network: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=512, # 400
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        if predict_senses:
            self.linear2senses = torch.nn.Linear(in_features=512, # 400
                                                 out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        distributed_batch_size = batchinput_tensor.shape[0]
        if not (distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            reshape_memories(distributed_batch_size, self)

        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()
        self.memory_cn.detach_()
        self.memory_hn_senses.detach_()
        self.memory_cn_senses.detach_()

        if batchinput_tensor.shape[1] > 1:
            time_instants = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[1], dim=1)
        else:
            time_instants = [batchinput_tensor]

        word_embeddings_ls = []
        currentglobal_nodestates_ls = []

        for batch_elements_at_t in time_instants:
            batch_elems_at_t = batch_elements_at_t.squeeze(dim=1)
            elems_at_t_ls = batch_elements_at_t.chunk(chunks=batch_elems_at_t.shape[0], dim=0)

            t_input_lts = [unpack_input_tensor(sample_tensor, self.grapharea_size) for sample_tensor in elems_at_t_ls]

            t_globals_indices_ls = [t_input_lts[b][0][0] for b in range(len(t_input_lts))]

            # Input signal n.1: the embedding of the current (global) word
            t_current_globals_indices_ls = [x_indices[0] for x_indices in t_globals_indices_ls]
            t_current_globals_indices = torch.stack(t_current_globals_indices_ls, dim=0)
            t_word_embeddings = self.X.index_select(dim=0, index=t_current_globals_indices)
            word_embeddings_ls.append(t_word_embeddings)

            # Input signal n.2: the node-state of the current global word - now with graph batching
            graph_batch_ls = []
            current_location_in_batchX_ls = []
            rows_to_skip = 0
            if self.include_globalnode_input:
                t_edgeindex_g_ls = [t_input_lts[b][0][1] for b in range(len(t_input_lts))]

                for i_sample in range(batch_elems_at_t.shape[0]):

                    sample_edge_index = t_edgeindex_g_ls[i_sample]
                    x_indices, sample_edge_index = lemmatize_node(t_globals_indices_ls[i_sample], sample_edge_index, self)
                    sample_x = self.X.index_select(dim=0, index=x_indices.squeeze())

                    currentword_location_in_batchX = rows_to_skip + current_location_in_batchX_ls[-1] \
                        if len(current_location_in_batchX_ls)>0 else 0
                    rows_to_skip = sample_x.shape[0]
                    current_location_in_batchX_ls.append(currentword_location_in_batchX)

                    sample_graph = Data(x=sample_x, edge_index=sample_edge_index)
                    graph_batch_ls.append(sample_graph)

                batch_graph = Batch.from_data_list(graph_batch_ls)
                x_attention_states = self.gat_globals(batch_graph.x, batch_graph.edge_index)
                t_currentglobal_node_states = x_attention_states.index_select(dim=0, index=torch.tensor(current_location_in_batchX_ls).to(torch.int64).to(CURRENT_DEVICE))
                currentglobal_nodestates_ls.append(t_currentglobal_node_states)

            # Input signal n.3: : the node-state of the current sense
            if self.include_sensenode_input:
                t_edgeindex_g_ls = [t_input_lts[b][0][1] for b in range(len(t_input_lts))]

                for i_sample in range(batch_elems_at_t.shape[0]):
                    sample_edge_index = t_edgeindex_g_ls[i_sample]
                    x_indices, sample_edge_index = lemmatize_node(t_globals_indices_ls[i_sample], sample_edge_index,
                                                                  self)
                    sample_x = self.X.index_select(dim=0, index=x_indices.squeeze())

                    currentword_location_in_batchX = rows_to_skip + current_location_in_batchX_ls[-1] \
                        if len(current_location_in_batchX_ls) > 0 else 0
                    rows_to_skip = sample_x.shape[0]
                    current_location_in_batchX_ls.append(currentword_location_in_batchX)

                    sample_graph = Data(x=sample_x, edge_index=sample_edge_index)
                    graph_batch_ls.append(sample_graph)

                batch_graph = Batch.from_data_list(graph_batch_ls)
                x_attention_states = self.gat_globals(batch_graph.x, batch_graph.edge_index)
                t_currentglobal_node_states = x_attention_states.index_select(dim=0, index=torch.tensor(
                    current_location_in_batchX_ls).to(torch.int64).to(CURRENT_DEVICE))
                currentglobal_nodestates_ls.append(t_currentglobal_node_states)
            else:
                currentsense_node_state = None

        word_embeddings = torch.stack(word_embeddings_ls, dim=0)
        global_nodestates = torch.stack(currentglobal_nodestates_ls, dim=0) if self.include_globalnode_input else None

        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                               [word_embeddings, global_nodestates])) #, currentsense_node_state]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        # - h_0 of shape (num_layers * num_directions, batch=1, hidden_size):
        #       tensor containing the initial hidden state for each element in the batch.
        main_rnn_out = None
        input = batch_input_signals
        for i in range(self.n_layers):
            layer_rnn = self.main_rnn_ls[i]
            layer_rnn.flatten_parameters()
            if self.model_type.upper() == "LSTM":
                main_rnn_out, (hidden_i, cells_i) = \
                    layer_rnn(input,select_layer_memory(self, i, layer_rnn))
                update_layer_memory(self, i, layer_rnn, hidden_i, cells_i)
            else: # GRU
                main_rnn_out, hidden_i = \
                    layer_rnn(input,select_layer_memory(self, i, layer_rnn))
                update_layer_memory(self, i, layer_rnn, hidden_i)

            input = main_rnn_out

        main_rnn_out = main_rnn_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)
        seq_len = batch_input_signals.shape[0]
        main_rnn_out = main_rnn_out.reshape(distributed_batch_size * seq_len, main_rnn_out.shape[2])

        # 2nd part of the architecture: predictions
        # globals
        logits_global = self.linear2global(main_rnn_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)

        # senses
        # line 1: GRU for senses + linear FF-NN to logits.
        if self.predict_senses:
            senses_rnn_output = None
            input = batch_input_signals
            for i in range(self.n_layers):
                layer_rnn = self.senses_rnn_ls[i]
                layer_rnn.flatten_parameters()
                if self.model_type.upper() == "LSTM":
                    senses_rnn_output, (hidden_i, cells_i) = \
                        layer_rnn(input, select_layer_memory(self, i, layer_rnn))
                    update_layer_memory(self, i, layer_rnn, hidden_i, cells_i)
                else:  # GRU
                    senses_rnn_output, hidden_i = \
                        layer_rnn(input, select_layer_memory(self, i, layer_rnn))
                    update_layer_memory(self, i, layer_rnn, hidden_i)

                senses_rnn_output = self.dropout(senses_rnn_output)
                input = senses_rnn_output

            senses_rnn_output = senses_rnn_output.reshape(distributed_batch_size * seq_len, senses_rnn_output.shape[2])

            logits_sense = self.linear2senses(senses_rnn_output)

            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
