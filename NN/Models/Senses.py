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

# ****** Auxiliary functions *******
def get_senseneighbours_of_k_globals(model, sample_k_indices):
    neighbours_of_k = torch.cat(
                    [AD.get_node_data(model.grapharea_matrix, i, model.grapharea_size, features_mask=(True,False,False))[0]
                     for i in sample_k_indices], dim=0)
    sense_neighbours_of_k = neighbours_of_k[neighbours_of_k < model.last_idx_senses]
    return sense_neighbours_of_k

def subtract_probability_mass_from_selected(softmax_selected_senses, delta_to_subtract):
    max_index_t = torch.argmax(softmax_selected_senses)
    prev_max_value = softmax_selected_senses[max_index_t]
    softmax_selected_senses[max_index_t].data = prev_max_value - delta_to_subtract
    return softmax_selected_senses

# *****************************

# Choose among the senses of the most likely 5 globals.
# Add the [the probability distribution over those] to [the distribution over the whole senses' vocabulary]...

class SelectK(torch.nn.Module):
    def __init__(self, data, grapharea_size, grapharea_matrix, num_k_globals, vocabulary_wordlist, include_globalnode_input, predict_senses,
                 batch_size, n_layers, n_hid_units):
        super(SelectK, self).__init__()
        init_model_parameters(self, data, grapharea_size, grapharea_matrix, vocabulary_wordlist,
                              include_globalnode_input, predict_senses,
                              batch_size, n_layers, n_hid_units, dropout_p=0)
        self.k=num_k_globals

        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        self.select_first_indices = Parameter(torch.tensor(list(range(n_hid_units))).to(torch.float32), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        # Input signals: current global’s word embedding (|| global's node state (|| sense’s node state) )
        self.concatenated_input_dim = self.dim_embs * (1 + int(include_globalnode_input))

        # This is overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers - 1, batch_size, int(self.hidden_size))),
                                          requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # GRU for globals - standard Language Model
        self.main_rnn_ls = torch.nn.ModuleList([
            torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else self.hidden_size,
                          hidden_size=self.hidden_size if i == n_layers - 1 else self.hidden_size, num_layers=1) for i in
            range(n_layers)])
        # GATs for the node-states
        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=self.dim_embs, out_channels=int(self.dim_embs / 4), heads=4)
        # GRU for senses
        if predict_senses:
            self.gru_senses = torch.nn.GRU(input_size=self.concatenated_input_dim, hidden_size=int(self.hidden_size), num_layers=n_layers - 1)

        # lemmatizer, we may use it for the globals or for the SelectK
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatize_term('init', self.lemmatizer)  # to establish LazyCorpusLoader and prevent a multi-thread crash

        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=self.hidden_size,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        if predict_senses:
            self.linear2senses = torch.nn.Linear(in_features=int(self.hidden_size),
                                                 out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        distributed_batch_size = batchinput_tensor.shape[0]
        if not (distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            new_num_hidden_state_elems = self.n_layers * distributed_batch_size * self.hidden_size
            self.memory_hn = Parameter(torch.reshape(self.memory_hn.flatten()[0:new_num_hidden_state_elems],
                                                     (self.n_layers, distributed_batch_size, self.hidden_size)),
                                       requires_grad=False)
            self.memory_hn_senses = Parameter(
                torch.reshape(self.memory_hn_senses.flatten()[
                              0:((self.n_layers - 1) * distributed_batch_size * int(self.hidden_size))],
                              (self.n_layers - 1, distributed_batch_size, int(self.hidden_size))),
                requires_grad=False)
            self.hidden_state_bsize_adjusted = True

        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()
        self.memory_hn_senses.detach_()

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
                    if x_indices_g.shape[0] <= 1:  # if we have an isolated node, that may be an inflected form ('said')
                        currentglobal_relative_X_idx = x_indices_g[0]
                        currentglobal_absolute_vocab_idx = currentglobal_relative_X_idx - self.last_idx_senses
                        word = self.vocabulary_wordlist[currentglobal_absolute_vocab_idx]
                        lemmatized_word = lemmatize_term(word, self.lemmatizer)
                        if lemmatized_word != word:  # ... (or a stopword, in which case we do not proceed further)
                            try:
                                lemmatized_word_absolute_idx = self.vocabulary_wordlist.index(lemmatized_word)
                                lemmatized_word_relative_idx = lemmatized_word_absolute_idx + self.last_idx_senses
                                (x_indices_g, edge_index_g, edge_type_g) = \
                                    AD.get_node_data(self.grapharea_matrix, lemmatized_word_relative_idx, self.grapharea_size)
                            except ValueError:
                                pass  # the lemmatized word was not found in the vocabulary.

                    x = self.X.index_select(dim=0, index=x_indices_g.squeeze())
                    x_attention_state = self.gat_globals(x, edge_index_g)
                    currentglobal_node_state = x_attention_state.index_select(dim=0,
                                                                              index=self.select_first_indices[0].to(
                                                                                  torch.int64))
                else:
                    currentglobal_node_state = None

                input_ls = list(filter(lambda signal: signal is not None,
                                       [currentword_embedding, currentglobal_node_state, currentsense_node_state]))
                input_signals = torch.cat(input_ls, dim=1)
                sequence_input_signals_ls.append(input_signals)

            sequence_input_signals = torch.cat(sequence_input_signals_ls, dim=0).unsqueeze(1)
            batch_input_signals_ls.append(sequence_input_signals)
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=1)

        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        # - h_0 of shape (num_layers * num_directions, batch=1, hidden_size):
        #       tensor containing the initial hidden state for each element in the batch.
        input = batch_input_signals
        input = batch_input_signals
        for i in range(self.n_layers):
            layer_rnn = self.main_rnn_ls[i]
            layer_rnn.flatten_parameters()
            main_gru_out, hidden_i = \
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
            main_gru_out = self.dropout(main_gru_out)
            input = main_gru_out

        main_gru_out = main_gru_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)
        seq_len = len(sequences_in_the_batch_ls[0][0])
        main_gru_out = main_gru_out.reshape(distributed_batch_size * seq_len, main_gru_out.shape[2])


        # 2nd part of the architecture: predictions
        # globals
        logits_global = self.linear2global(main_gru_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)
        # senses
        if self.predict_senses:
            # line 1: GRU for senses + linear FF-NN to logits.
            self.gru_senses.flatten_parameters()
            senses_gru_out, hidden_s =self.gru_senses(batch_input_signals, self.memory_hn_senses)
            self.memory_hn_senses.data.copy_(hidden_s.clone())  # store h in memory
            senses_gru_out = senses_gru_out.reshape(distributed_batch_size * seq_len, senses_gru_out.shape[2])
            logits_sense = self.linear2senses(senses_gru_out)

            # line 2: select senses of the k most likely globals
            k_globals_indices = logits_global.sort(descending=True).indices[:, 0:self.k]

            senses_softmax = torch.ones((distributed_batch_size * seq_len, self.last_idx_senses)).to(CURRENT_DEVICE)
            epsilon = 10 ** (-8)
            senses_softmax = epsilon * senses_softmax  # base probability value for non-selected senses
            i_senseneighbours_mask = torch.zeros(size=(distributed_batch_size * seq_len, self.last_idx_senses)).to(torch.bool).to(CURRENT_DEVICE)

            sample_k_indices_lls_relative = k_globals_indices.tolist()

            for s in range(distributed_batch_size * seq_len):
                k_globals_relative_indices = sample_k_indices_lls_relative[s]
                k_globals_words = [self.vocabulary_wordlist[global_relative_idx] for global_relative_idx in
                                   k_globals_relative_indices]
                k_globals_lemmatized = [lemmatize_term(word, self.lemmatizer) for word in k_globals_words]
                lemmatized_indices = [
                    Utils.word_to_vocab_index(lemmatized_word, self.vocabulary_wordlist) + self.last_idx_senses for
                    lemmatized_word in k_globals_lemmatized]
                sense_neighbours_t = get_senseneighbours_of_k_globals(self, lemmatized_indices)
                if sense_neighbours_t.shape[0] == 0:  # no senses found, even lemmatizing. Ignore current entry
                    senses_softmax[s] = torch.tensor(1 / self.last_idx_senses).to(CURRENT_DEVICE)
                    continue

                # standard procedure: get the logits of the senses of the most likely globals,
                # apply a softmax only over them, and then assign an epsilon probability to the other senses
                sample_logits_senses = logits_sense.index_select(dim=0, index=self.select_first_indices[s].to(torch.int64)).squeeze()
                logits_selected_senses = sample_logits_senses.index_select(dim=0, index=sense_neighbours_t)
                softmax_selected_senses = tfunc.softmax(input=logits_selected_senses, dim=0)

                for i in range(len(sense_neighbours_t)):
                    i_senseneighbours_mask[s,sense_neighbours_t[i]]=True

                quantity_to_subtract_from_selected = epsilon * (self.last_idx_senses - len(sense_neighbours_t))
                softmax_selected_senses = subtract_probability_mass_from_selected(softmax_selected_senses, quantity_to_subtract_from_selected)
                senses_softmax[s].masked_scatter_(mask=i_senseneighbours_mask[s].data.clone(), source=softmax_selected_senses)

            predictions_senses = torch.log(senses_softmax)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses