import torch
from torch_geometric.nn import GATConv
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
import torch.nn.functional as tfunc
from GNN.Models.Common import unpack_input_tensor, init_model_parameters, lemmatize_node
from torch.nn.parameter import Parameter
import logging
import nltk
from PrepareKBInput.LemmatizeNyms import lemmatize_term

# Parameters: the x and edge_index of the grapharea of 1 node; a Lemmatizer; the graph we are operating on.
# Returns: if the node's word can be lemmatized: the x and edge_index of the lemmatized token (e.g. 'said' -> 'say')
#                                          else: the parameters x and edge_index of the original node, unchanged.

class RNN(torch.nn.Module):

    def __init__(self, model_type, data, grapharea_size, grapharea_matrix, vocabulary_wordlist, include_globalnode_input, include_sensenode_input, predict_senses,
                 batch_size, n_layers, n_hid_units):
        super(RNN, self).__init__()
        self.model_type = model_type # can be "LSTM" or "GRU"
        init_model_parameters(self, data, grapharea_size, grapharea_matrix, vocabulary_wordlist,
                          include_globalnode_input, include_sensenode_input, predict_senses,
                          batch_size, n_layers, n_hid_units, dropout_p=0)

        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_indices = Parameter(torch.tensor(list(range(n_hid_units))).to(torch.float32), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        # Input signals: current global’s word embedding (|| global's node state (|| sense’s node state) )
        self.concatenated_input_dim = self.dim_embs * (1 + int(include_globalnode_input) + int(include_sensenode_input))

        # This is overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_cn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers-1, batch_size, int(self.hidden_size))), requires_grad=False)
        self.memory_cn_senses = Parameter(torch.zeros(size=(n_layers-1, batch_size, int(self.hidden_size))), requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # RNN for globals - standard Language Model
        if self.model_type.upper() == 'LSTM':
            self.main_rnn_ls = torch.nn.ModuleList([
                torch.nn.LSTM(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                              hidden_size=n_hid_units if i == n_layers - 1 else n_hid_units, num_layers=1) for i in range(n_layers)])
        else: # GRU
            self.main_rnn_ls = torch.nn.ModuleList([
                torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                              hidden_size=n_hid_units if i == n_layers - 1 else n_hid_units, num_layers=1) for
                i in range(n_layers)])

        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=self.dim_embs, out_channels=int(self.dim_embs / 4), heads=4)#, node_dim=1)
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
            lemmatize_term('init', self.lemmatizer)# to establish LazyCorpusLoader and prevent a multi-thread crash
        if self.include_sensenode_input:
            self.gat_senses = GATConv(in_channels=self.dim_embs, out_channels=int(self.dim_embs / 4), heads=4)
        # RNN for senses
        if predict_senses:
            if self.model_type.upper() == 'LSTM':
                self.rnn_senses = torch.nn.LSTM(input_size=self.concatenated_input_dim, hidden_size=int(self.hidden_size), num_layers=n_layers - 1)
            else: # GRU
                self.rnn_senses = torch.nn.GRU(input_size=self.concatenated_input_dim,
                                                hidden_size=int(self.hidden_size), num_layers=n_layers - 1)

        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=int(self.hidden_size),
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        if predict_senses:
            self.linear2senses = torch.nn.Linear(in_features=int(self.hidden_size),
                                                 out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        distributed_batch_size = batchinput_tensor.shape[0]
        if not(distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            new_num_hidden_state_elems = self.n_layers * distributed_batch_size * self.hidden_size
            self.memory_hn = Parameter(torch.reshape(self.memory_hn.flatten()[0:new_num_hidden_state_elems],
                                                     (self.n_layers, distributed_batch_size, self.hidden_size)), requires_grad=False)
            self.memory_cn = Parameter(torch.reshape(self.memory_cn.flatten()[0:new_num_hidden_state_elems],
                                                     (self.n_layers, distributed_batch_size, self.hidden_size)),
                                       requires_grad=False)
            self.memory_hn_senses = Parameter(
                            torch.reshape(self.memory_hn_senses.flatten()[0:((self.n_layers-1) * distributed_batch_size * int(self.hidden_size))],
                                          (self.n_layers-1, distributed_batch_size, int(self.hidden_size))),
                            requires_grad=False)
            self.memory_cn_senses = Parameter(
                torch.reshape(self.memory_hn_senses.flatten()[
                              0:((self.n_layers-1) * distributed_batch_size * int(self.hidden_size))],
                              (self.n_layers-1, distributed_batch_size, int(self.hidden_size))),
                requires_grad=False)
            self.hidden_state_bsize_adjusted=True

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
            logging.debug("shapes in t_globals_indices_ls=" + str([t_globals.shape for t_globals in t_globals_indices_ls]))

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

                    sample_x = self.X.index_select(dim=0, index=t_globals_indices_ls[i_sample].squeeze())
                    sample_edge_index = t_edgeindex_g_ls[i_sample]
                    lemmatize_node(t_globals_indices_ls[i_sample], sample_edge_index, self)

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
                # in development
                pass
                #
                # t_senses_indices_ls = [t_input_lts[b][1][0] for b in range(len(t_input_lts))]
                # logging.info("shapes in t_senses_indices_ls=" + str([t_senses.shape for t_senses in t_senses_indices_ls]))
                # t_edgeindex_g_ls = [t_input_lts[b][1][1] for b in range(len(t_input_lts))]
                #
                #
                # if len(t_senses_indices_ls[t_senses_indices_ls != 0] == 0):  # no sense was specified
                #     currentsense_node_state = self.embedding_zeros
                # else:  # sense was specified
                #     pass
                #     # x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
                #     # sense_attention_state = self.gat_senses(x_s, edge_index_s)
                #     # currentsense_node_state = sense_attention_state.index_select(dim=0,
                #     #                                                              index=self.select_first_indices[
                #     #                                                                  0].to(torch.int64))
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
                    layer_rnn(input,
                              (self.memory_hn.index_select(dim=0, index=self.select_first_indices[i].to(torch.int64)).
                               index_select(dim=2,index=self.select_first_indices[0:layer_rnn.hidden_size].to(torch.int64)),
                               self.memory_cn.index_select(dim=0, index=self.select_first_indices[i].to(torch.int64)).
                               index_select(dim=2,index=self.select_first_indices[0:layer_rnn.hidden_size].to(torch.int64))))
                cells_i_forcopy = cells_i.index_select(dim=2,
                                                       index=self.select_first_indices[0:layer_rnn.hidden_size].to(
                                                           torch.int64))
                cells_i_forcopy = tfunc.pad(cells_i_forcopy,
                                            pad=[0, self.memory_hn.shape[2] - layer_rnn.hidden_size]).squeeze()
                self.memory_cn[i].data.copy_(cells_i_forcopy.clone())

            else: # GRU
                main_rnn_out, hidden_i = \
                    layer_rnn(input,
                              self.memory_hn.index_select(dim=0, index=self.select_first_indices[i].to(torch.int64)).
                               index_select(dim=2,
                                            index=self.select_first_indices[0:layer_rnn.hidden_size].to(torch.int64)))
                hidden_i_forcopy = hidden_i.index_select(dim=2,
                                                         index=self.select_first_indices[0:layer_rnn.hidden_size].to(
                                                             torch.int64))
                hidden_i_forcopy = tfunc.pad(hidden_i_forcopy,
                                             pad=[0, (self.memory_hn.shape[2] - layer_rnn.hidden_size)]).squeeze()
                self.memory_hn[i].data.copy_(hidden_i_forcopy.clone())  # store h in memory
            main_rnn_out = self.dropout(main_rnn_out)
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
            self.rnn_senses.flatten_parameters()
            if self.model_type.upper() == "LSTM":
                senses_rnn_out, (hidden_s, cells_s) = self.rnn_senses(batch_input_signals, (self.memory_hn_senses, self.memory_cn_senses))
                self.memory_cn_senses.data.copy_(hidden_s.clone())  # store h in memory
            else: # GRU
                senses_rnn_out, hidden_s = self.rnn_senses(batch_input_signals, self.memory_hn_senses)

            self.memory_hn_senses.data.copy_(hidden_s.clone())  # store h in memory
            senses_rnn_out = senses_rnn_out.reshape(distributed_batch_size * seq_len, senses_rnn_out.shape[2])
            logits_sense = self.linear2senses(senses_rnn_out)

            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
