import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop #, ForwardWithDrop
import os, sys
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv
import nltk


# allowing for use of tools in the parent folder
sys.path.append(os.path.join(os.getcwd(), '..', ''))
from PrepareKBInput.LemmatizeNyms import lemmatize_term


class AWD(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, graph_dataobj, variant_flags_dict, my_vocabulary_wordlist,
                 dropout=0.5, dropouth=0.5, dropouti=0.5,
                 dropoute=0.1, wdrop=0, tie_weights=False):
        super(AWD, self).__init__()

        # added by me

        self.my_vocabulary_wordlist = my_vocabulary_wordlist
        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(graph_dataobj.x.clone().detach(), requires_grad=True)
        self.d_external_inp = self.X.shape[1]
        self.select_first_indices = Parameter(torch.tensor(list(range(ninp))).to(torch.float32), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.d_external_inp)), requires_grad=False)

        self.variant_flags_dict = variant_flags_dict # dictionary with my options for this particular model

        if variant_flags_dict['include_globalnode_input']:
            self.gat_globals = GATConv(in_channels=self.dim_embs, out_channels=int(self.dim_embs / 4), heads=4)
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
            lemmatize_term('init', self.lemmatizer)  # to establish LazyCorpusLoader and prevent a multi-thread crash
        if variant_flags_dict['include_sensenode_input']:
            self.gat_senses = GATConv(in_channels=self.dim_embs, out_channels=int(self.dim_embs / 4), heads=4)

        self.nhid = nhid
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp+self.d_external_inp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp+self.d_external_inp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        seq_len = input.shape[0]
        bsz = input.shape[1]

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []

        # we can add another input signal for the first layer
        # either the simple FastText embedding for the same word in the vocabulary (as a baseline)
        # or the state of the word's global node in the KB graph
        additional_input_signal_ls = []

        word_embeddings_across_batches_ls = []
        for t in range(seq_len):
            if not (self.variant_flags_dict['include_globalnode_input']):
                time_t_word_embeddings = self.X.index_select(dim=0, index=input[t, :])
                word_embeddings_across_batches_ls.append(time_t_word_embeddings)
        additional_input_signal = torch.cat(word_embeddings_across_batches_ls, dim=0)
        additional_input_signal = nn.functional.dropout(input=additional_input_signal, p=self.dropoute, training=self.training, inplace=False)
        dropout = self.dropoute if self.training else 0

        # for t in range(seq_len):
        #     for b in range(bsz):
        #         if not (self.variant_flags_dict['include_globalnode_input']):
        #             currentword_embedding = self.X.index_select(dim=0, index=input[t,b]) # just use the FastText word embedding
        #             additional_input_signal_ls.append(currentword_embedding)
        #         else:
        #             pass
                    # # lemmatization
                    # if x_indices_g.shape[
                    #     0] <= 1:  # if we have an isolated node, that may be an inflected form ('said')...
                    #     currentglobal_relative_X_idx = x_indices_g[0]
                    #     currentglobal_absolute_vocab_idx = currentglobal_relative_X_idx - self.last_idx_senses
                    #     word = self.vocabulary_wordlist[currentglobal_absolute_vocab_idx]
                    #     lemmatized_word = lemmatize_term(word, self.lemmatizer)
                    #     if lemmatized_word != word:  # ... (or a stopword, in which case we do not proceed further)
                    #         try:
                    #             lemmatized_word_absolute_idx = self.vocabulary_wordlist.index(lemmatized_word)
                    #             lemmatized_word_relative_idx = lemmatized_word_absolute_idx + self.last_idx_senses
                    #             (x_indices_g, edge_index_g, edge_type_g) = \
                    #                 AD.get_node_data(self.grapharea_matrix, lemmatized_word_relative_idx,
                    #                                  self.grapharea_size)
                    #         except ValueError:
                    #             pass  # the lemmatized word was not found in the vocabulary.
                    #
                    # x = self.X.index_select(dim=0, index=x_indices_g.squeeze())
                    # x_attention_state = self.gat_globals(x, edge_index_g)
                    # currentglobal_node_state = x_attention_state.index_select(dim=0,
                    #                                                           index=self.select_first_indices[0].to(
                    #                                                               torch.int64))

        additional_contribution = additional_input_signal.view(size=(seq_len,bsz, self.d_external_inp))

        for l, rnn in enumerate(self.rnns):

            rnn.module.flatten_parameters()  # * now working
            current_input = raw_output
            # my modification (insertion of the additional input signal by concatenation):
            if l == 0:
                raw_output = torch.cat([current_input, additional_contribution], dim=2)
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
