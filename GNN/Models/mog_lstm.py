import torch
import torch.nn as nn
import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Graph.Adjacencies as AD
from GNN.Models.Common import unpack_input_tensor
from GNN.Models.Common import init_model_parameters
from Utils import DEVICE
from torch.nn.parameter import Parameter
import logging
import GNN.ExplorePredictions as EP
import Utils
import nltk
from PrepareKBInput.LemmatizeNyms import lemmatize_term


# ******* Modified and expanded upon,
# ******* from the PyTorch implementation of Mogrifier-LSTM, at : https://github.com/fawazsammani/mogrifier-lstm-pytorch

class MogrifierLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogrifierLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r

    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i + 1) % 2 == 0:
                h = (2 * torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2 * torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct

# *********************

# ********* Hyperparameters from the paper: "Mogrifier LSTM by G.Melis et al., 2020"

# input_size = 512
# hidden_size = 512
# vocab_size = 30
# batch_size = 4
# lr = 3e-3
# mogrify_steps = 5  # 5 steps give optimal performance according to the paper
# dropout = 0.5  # for simplicity: input dropout and output_dropout are 0.5. See appendix B in the paper for exact values
# tie_weights = True  # in the paper, embedding weights and output weights are tied
# betas = (0, 0.999)  # in the paper the momentum term in Adam is ignored
# weight_decay = 2.5e-4  # weight decay is around this value, see appendix B in the paper
# clip_norm = 10  # paper uses cip_norm of 10
#
# model = Model(input_size, hidden_size, mogrify_steps, vocab_size, tie_weights, dropout)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e-08, weight_decay=weight_decay)
#
# # seq of shape (batch_size, max_words)
# seq = torch.LongTensor([[8, 29, 18, 1, 17, 3, 26, 6, 26, 5],
#                         [8, 28, 15, 12, 13, 2, 26, 16, 20, 0],
#                         [15, 4, 27, 14, 29, 28, 14, 1, 0, 0],
#                         [20, 22, 29, 22, 23, 29, 0, 0, 0, 0]])
#
# outputs, hidden_states = model(seq)
# print(outputs.shape)
# print(hidden_states.shape)

# **** Modified from GitHub: "Here we provide an example of a model with two-layer Mogrifier LSTM."

class MOG_LSTM(nn.Module):
    def __init__(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_wordlist, mogrify_steps,
                 include_globalnode_input, include_sensenode_input, predict_senses,
                 batch_size, n_layers, n_hid_units):
        super(MOG_LSTM, self).__init__()
        # Initializing model parameters
        init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_wordlist,
                              include_globalnode_input, include_sensenode_input, predict_senses,
                              batch_size, n_layers, n_hid_units, dropout_p=0.1)

        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(graph_dataobj.x.clone().detach(), requires_grad=True)
        self.select_first_indices = Parameter(torch.tensor(list(range(n_hid_units))).to(torch.float32), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        # Input signals: current global’s word embedding (|| global's node state (|| sense’s node state) )
        self.concatenated_input_dim = self.dim_embs * (1 + int(include_globalnode_input) + int(include_sensenode_input))

        # This is overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_cn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers-1, batch_size, n_hid_units), requires_grad=False))
        self.memory_cn_senses = Parameter(torch.zeros(size=(n_layers-1, batch_size, n_hid_units), requires_grad=False))
        self.hidden_state_bsize_adjusted = False

        # Mogrifier LSTM for globals - standard Language Model
        self.mogrify_steps = mogrify_steps
        self.main_moglstm_layer1 = MogrifierLSTMCell(self.concatenated_input_dim, self.hidden_size, self.mogrify_steps)
        self.main_moglstm_layer2 = MogrifierLSTMCell(self.hidden_size, self.hidden_size, self.mogrify_steps)
        self.main_rnn_ls = torch.nn.ModuleList([self.main_moglstm_layer1, self.main_moglstm_layer2])

        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=self.dim_embs, out_channels=int(self.dim_embs / 4), heads=4)
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
            lemmatize_term('init', self.lemmatizer) # to establish LazyCorpusLoader and prevent a multi-thread crash
        if self.include_sensenode_input:
            self.gat_senses = GATConv(in_channels=self.dim_embs, out_channels=int(self.dim_embs / 4), heads=4)
        # Mogrifier LSTM for senses
        if predict_senses:
            self.moglstm_senses_layer1 = MogrifierLSTMCell(self.concatenated_input_dim, self.hidden_size,
                                                         self.mogrify_steps)
            self.moglstm_senses_layer2 = MogrifierLSTMCell(self.hidden_size, self.hidden_size,
                                                         self.mogrify_steps)

        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=self.hidden_size,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        if predict_senses:
            self.linear2senses = torch.nn.Linear(in_features=self.hidden_size,
                                                 out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        distributed_batch_size = batchinput_tensor.shape[0]
        if not(distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            new_num_hidden_state_elems = self.n_layers * distributed_batch_size * self.n_units
            self.memory_hn = Parameter(torch.reshape(self.memory_hn.flatten()[0:new_num_hidden_state_elems],
                                           (self.n_layers, distributed_batch_size, self.n_units)), requires_grad=False)
            self.memory_cn = Parameter(torch.reshape(self.memory_cn.flatten()[0:new_num_hidden_state_elems],
                                                     (self.n_layers, distributed_batch_size, self.n_units)),
                                       requires_grad=False)
            self.memory_hn_senses = Parameter(
                            torch.reshape(self.memory_hn_senses.flatten()[0:((self.n_layers-1) *distributed_batch_size * self.hidden_size)],
                                          (self.n_layers-1, distributed_batch_size, self.hidden_size)),
                            requires_grad=False)
            self.memory_cn_senses = Parameter(
                torch.reshape(self.memory_hn_senses.flatten()[
                              0:((self.n_layers-1) * distributed_batch_size * self.hidden_size)],
                              (self.n_layers-1, distributed_batch_size, self.hidden_size)),
                requires_grad=False)
            self.hidden_state_bsize_adjusted=True

        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()
        self.memory_cn.detach_()
        self.memory_hn_senses.detach_()
        self.memory_cn_senses.detach_()

        if batchinput_tensor.shape[0] > 1:
            sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
        else:
            sequences_in_the_batch_ls = [batchinput_tensor]

        batch_input_signals_ls = []

        for padded_sequence in sequences_in_the_batch_ls:
            padded_sequence = padded_sequence.squeeze(dim=0)
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [unpack_input_tensor(sample_tensor, self.grapharea_size) for sample_tensor in padded_sequence]

            sequence_input_signals_ls = []

            for ((x_indices_g, edge_index_g, edge_type_g), (x_indices_s, edge_index_s, edge_type_s)) in sequence_lts:
                # Input signal n.1: the embedding of the current (global) word
                currentword_embedding = self.X.index_select(dim=0, index=x_indices_g[0])

                # Input signal n.2: the node-state of the current global word
                if self.include_globalnode_input:
                    # lemmatization
                    if x_indices_g.shape[0]<=1: # if we have an isolated node, that may be an inflected form ('said')...
                        currentglobal_relative_X_idx = x_indices_g[0]
                        currentglobal_absolute_vocab_idx = currentglobal_relative_X_idx - self.last_idx_senses
                        word = self.vocabulary_wordlist[currentglobal_absolute_vocab_idx]
                        lemmatized_word = lemmatize_term(word, self.lemmatizer)
                        if lemmatized_word != word: # ... (or a stopword, in which case we do not proceed further)
                            try:
                                lemmatized_word_absolute_idx = self.vocabulary_wordlist.index(lemmatized_word)
                                lemmatized_word_relative_idx = lemmatized_word_absolute_idx + self.last_idx_senses
                                (x_indices_g, edge_index_g, edge_type_g) = \
                                    AD.get_node_data(self.grapharea_matrix, lemmatized_word_relative_idx, self.grapharea_size)
                            except ValueError:
                                pass # the lemmatized word was not found in the vocabulary.

                    x = self.X.index_select(dim=0, index=x_indices_g.squeeze())
                    x_attention_state = self.gat_globals(x, edge_index_g)
                    currentglobal_node_state = x_attention_state.index_select(dim=0, index=self.select_first_indices[0].to(torch.int64))
                else:
                    currentglobal_node_state = None

                # Input signal n.3: the node-state of the current sense; + concatenating the input signals
                if self.include_sensenode_input:
                    if len(x_indices_s[x_indices_s != 0] == 0):  # no sense was specified
                        currentsense_node_state = self.embedding_zeros
                    else:  # sense was specified
                        x_s = self.X.index_select(dim=0, index=x_indices_s.squeeze())
                        sense_attention_state = self.gat_senses(x_s, edge_index_s)
                        currentsense_node_state = sense_attention_state.index_select(dim=0,index=self.select_first_indices[0].to(torch.int64))
                else:
                    currentsense_node_state = None
                input_ls = list(filter( lambda signal: signal is not None,
                                        [currentword_embedding, currentglobal_node_state, currentsense_node_state]))
                input_signals = torch.cat(input_ls, dim=1)
                sequence_input_signals_ls.append(input_signals)

            sequence_input_signals = torch.cat(sequence_input_signals_ls, dim=0).unsqueeze(1)
            batch_input_signals_ls.append(sequence_input_signals)
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=1)

        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        # - h_0 of shape (num_layers * num_directions, batch=1, hidden_size):
        #       tensor containing the initial hidden state for each element in the batch.

        main_rnn_out = None
        input = batch_input_signals
        for i in range(self.n_layers):
            layer_rnn = self.main_rnn_ls[i]
            # layer_rnn.flatten_parameters()
            main_rnn_out, (hidden_i, cells_i) = \
                    layer_rnn(input,
                              (self.memory_hn.index_select(dim=0, index=self.select_first_indices[i].to(torch.int64)),
                               self.memory_cn.index_select(dim=0, index=self.select_first_indices[i].to(torch.int64))))
            self.memory_cn[i].data.copy_(cells_i.clone())
            self.memory_hn[i].data.copy_(hidden_i.clone())  # store h in memory

            main_rnn_out = self.dropout(main_rnn_out)
            input = main_rnn_out

        main_rnn_out = main_rnn_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)
        seq_len = len(sequences_in_the_batch_ls[0][0])
        main_rnn_out = main_rnn_out.reshape(distributed_batch_size * seq_len, main_rnn_out.shape[2])

        # 2nd part of the architecture: predictions
        # globals
        logits_global = self.linear2global(main_rnn_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)
        # senses
        # line 1: GRU for senses + linear FF-NN to logits.
        if self.predict_senses:
            self.rnn_senses.flatten_parameters()
            senses_rnn_out, (hidden_s, cells_s) = self.rnn_senses(batch_input_signals, (self.memory_hn_senses, self.memory_cn_senses))
            self.memory_cn_senses.data.copy_(cells_s.clone())  # store h in memory
            self.memory_hn_senses.data.copy_(hidden_s.clone())  # store h in memory
            senses_rnn_out = senses_rnn_out.reshape(distributed_batch_size * seq_len, senses_rnn_out.shape[2])
            logits_sense = self.linear2senses(senses_rnn_out)

            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses