import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Graph.Adjacencies as AD
from GNN.Models.Common import unpack_input_tensor
from GNN.Models.Common import SelfAttention
from Utils import DEVICE
from torch.nn.parameter import Parameter
import logging
import GNN.ExplorePredictions as EP
import Utils
import nltk
from PrepareKBInput.LemmatizeNyms import lemmatize_term


# Baseline: GRUs only.
class GRU_base2(torch.nn.Module):

    def __init__(self, data, grapharea_size, grapharea_matrix, vocabulary_wordlist, include_globalnode_input, include_sensenode_input, predict_senses,
                 batch_size, n_layers, n_units):
        super(GRU_base2, self).__init__()
        self.grapharea_matrix = grapharea_matrix
        self.vocabulary_wordlist = vocabulary_wordlist
        self.include_globalnode_input = include_globalnode_input
        self.include_sensenode_input = include_sensenode_input
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

        # Input signals: current global’s word embedding (|| global's node state (|| sense’s node state) )
        self.concatenated_input_dim = self.d * (1 + int(include_globalnode_input) + int(include_sensenode_input))

        # This is overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_units//2))), requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # GRU for globals - standard Language Model
        self.main_gru = torch.nn.GRU(input_size=self.concatenated_input_dim, hidden_size=n_units, num_layers=n_layers)
        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=self.d, out_channels=int(self.d/4), heads=4)
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
            lemmatize_term('init', self.lemmatizer)# to establish LazyCorpusLoader and prevent a multi-thread crash
        if self.include_sensenode_input:
            self.gat_senses = GATConv(in_channels=self.d, out_channels=int(self.d/4), heads=4)
        # GRU for senses
        if predict_senses:
            self.gru_senses = torch.nn.GRU(input_size=self.concatenated_input_dim, hidden_size=int(n_units//2), num_layers=n_layers)

        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=n_units,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        if predict_senses:
            self.linear2senses = torch.nn.Linear(in_features=int(n_units//2),
                                                 out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        distributed_batch_size = batchinput_tensor.shape[0]
        if not(distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            new_num_hidden_state_elems = self.n_layers * distributed_batch_size * self.n_units
            self.memory_hn = Parameter(torch.reshape(self.memory_hn.flatten()[0:new_num_hidden_state_elems],
                                           (self.n_layers, distributed_batch_size, self.n_units)), requires_grad=False)
            self.memory_hn_senses = Parameter(
                            torch.reshape(self.memory_hn_senses.flatten()[0:((self.n_layers) *distributed_batch_size * int(self.n_units//2))],
                                          (self.n_layers, distributed_batch_size, int(self.n_units//2))),
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
            padded_sequence = padded_sequence.squeeze(dim=0)
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

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
                                    AD.get_node_data(self.grapharea_matrix, lemmatized_word_relative_idx, self.N)
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
        self.main_gru.flatten_parameters()
        main_gru_out, hidden_n = self.main_gru(batch_input_signals, self.memory_hn)
        self.memory_hn.data.copy_(hidden_n.clone())  # store h in memory
        main_gru_out = main_gru_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)
        seq_len = len(sequences_in_the_batch_ls[0][0])
        main_gru_out = main_gru_out.reshape(distributed_batch_size * seq_len, main_gru_out.shape[2])

        # 2nd part of the architecture: predictions
        # globals
        logits_global = self.linear2global(main_gru_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)
        # senses
        # line 1: GRU for senses + linear FF-NN to logits.
        if self.predict_senses:
            self.gru_senses.flatten_parameters()
            senses_gru_out, hidden_s =self.gru_senses(batch_input_signals, self.memory_hn_senses)
            self.memory_hn_senses.data.copy_(hidden_s.clone())  # store h in memory
            senses_gru_out = senses_gru_out.reshape(distributed_batch_size * seq_len, senses_gru_out.shape[2])
            logits_sense = self.linear2senses(senses_gru_out)

            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
