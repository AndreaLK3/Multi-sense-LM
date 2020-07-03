import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop  #, ForwardWithDrop
from weight_drop import WeightDrop  #, ForwardWithDrop
import os, sys
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
import nltk


# allowing for use of tools in the parent folder
sys.path.append(os.path.join(os.getcwd(), '..', ''))
from PrepareKBInput.LemmatizeNyms import lemmatize_term
from GNN.Models.Common import lemmatize_node
from Graph.Adjacencies import get_node_data

# Idea: Using 2 models in parallel:
# 1) Standard AWD-LSTM using its own d400 embeddings
# 2) Modified, using the d300 FastText embeddings (and later the GAT input, etc.).
# Combine them by: using the weighted & learned average of the softmax (or logits depending on how it works numerically)
# This weight can be 1 number,
# determined by, for instance, a 1-layer LSTM operating on the concatenation of the last encoding 400+300


# Auxiliary function, to make a 2D mask for a scatter update. It may be moved.
def make_2D_mask(indices_rows_to_include, max_vocab_index, dim_input):
    sorted_indices = indices_rows_to_include.sort(dim=0, descending=False).values.unique()
    mask1D_ls = [False] * max_vocab_index
    for i in range(sorted_indices.shape[0]):
        mask1D_ls[sorted_indices[i]]=True
    mask1D = torch.tensor(mask1D_ls)
    mask11D = mask1D.unsqueeze(dim=1)
    mask2D = mask11D.expand((mask1D.shape[0], dim_input))
    return mask2D



class AWD_ensemble(nn.Module):

    def __init__(self, AWD_base, AWD_modified, batch_size):
        super(AWD_ensemble, self).__init__()
        self.AWD_base = AWD_base
        self.AWD_modified = AWD_modified
        self.ninp = self.AWD_base.ninp # placeholder, we are not splitting the softmax now

        self.concatenated_encoding_dim = self.AWD_base.ninp + self.AWD_modified.ninp
        self.A = nn.LSTM(input_size=self.concatenated_encoding_dim, hidden_size=1, num_layers=1, bias=True) # layer to the coefficient (a) used to combine the logsoftmax
        self.memory_a_hidden = (torch.zeros(size=(1,batch_size,1)), torch.zeros(size=(1, batch_size, 1))) # used in splitcross.py > forward_ensemble(...)

    def forward(self, input, hidden, return_h=False):

        if not(return_h):
            result_base, hidden_base = self.AWD_base.forward(input, hidden[0], return_h)
            result_mod, hidden_mod = self.AWD_modified.forward(input, hidden[1], return_h)
            return ((result_base, result_mod), (hidden_base, hidden_mod))
        else:
            result_base, hidden_base, raw_outputs_base, outputs_base = self.AWD_base.forward(input, hidden[0], return_h)
            result_mod, hidden_mod, raw_outputs_mod, outputs_mod = self.AWD_modified.forward(input, hidden[1], return_h)
            return ((result_base, result_mod), (hidden_base, hidden_mod), (raw_outputs_base, raw_outputs_mod), (outputs_base, outputs_mod))

    def init_hidden(self, bsz):
        hidden_base = self.AWD_base.init_hidden(bsz)
        hidden_modified = self.AWD_modified.init_hidden(bsz)
        return (hidden_base, hidden_modified)


class AWD_modified(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, nhid, nlayers, graph_dataobj, variant_flags_dict,
                 my_vocabulary_wordlist, grapharea_matrix, grapharea_size,
                 dropout=0.5, dropouth=0.5, dropouti=0.5,
                 dropoute=0.1, wdrop=0, tie_weights=False):
        super(AWD_modified, self).__init__()

        # added by me
        self.vocabulary_wordlist = my_vocabulary_wordlist
        self.grapharea_matrix = grapharea_matrix
        self.grapharea_size = grapharea_size
        # The embeddings matrix for: senses, globals, definitions, examples
        self.X = Parameter(graph_dataobj.x.clone().detach(), requires_grad=True)
        self.d_inp = self.X.shape[1]
        self.last_idx_senses = graph_dataobj.node_types.tolist().index(1)
        self.last_idx_globals = graph_dataobj.node_types.tolist().index(2)
        self.select_first_indices = Parameter(torch.tensor(list(range(self.d_inp))).to(torch.float32), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.d_inp)), requires_grad=False)

        self.variant_flags_dict = variant_flags_dict  # dictionary with my options for this particular model

        if variant_flags_dict['include_globalnode_input']:
            self.gat_globals = GATConv(in_channels=self.d_inp, out_channels=int(self.d_inp / 4), heads=4)
            self.lemmatizer = nltk.stem.WordNetLemmatizer()
            lemmatize_term('init', self.lemmatizer)  # to establish LazyCorpusLoader and prevent a multi-thread crash
        if variant_flags_dict['include_sensenode_input']: # not fully implemented yet
            self.gat_senses = GATConv(in_channels=self.dim_embs, out_channels=int(self.dim_embs / 4), heads=4)

        self.ninp = self.d_inp
        self.nhid = nhid
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.d_encoding = self.d_inp * (1 + sum([int(variant_flags_dict[flag]) for flag in variant_flags_dict]))
        self.encoder = nn.Embedding(ntoken, self.d_encoding)
        self.ntokens_vocab = ntoken
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(self.d_inp if l == 0 else nhid, nhid if l != nlayers - 1 else (self.d_encoding if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(self.d_inp if l == 0 else nhid, nhid if l != nlayers - 1 else self.d_encoding, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        # elif rnn_type == 'QRNN':
        #     from torchqrnn import QRNNLayer
        #     self.rnns = [QRNNLayer(input_size=self.d_inp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (self.d_inp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
        #     for rnn in self.rnns:
        #         rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights:
        if tie_weights:
            # if nhid != ninp:
            #     raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights(variant_flags_dict, self.X)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self, variant_flags_dict, X):
        if not (variant_flags_dict['include_globalnode_input']):
            # initializing the nn.Embeddings object with the FastText embeddings, that we have at the start in X
            self.encoder.weight.data = X[self.last_idx_senses:self.last_idx_globals,:]
        else:
            self.encoder.weight.data = (X[self.last_idx_senses:self.last_idx_globals,:]).repeat([1,2]) # (33278 x 600)
        self.decoder.bias.data.fill_(0)


    def forward(self, input, hidden, return_h=False):
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())
        seq_len = input.shape[0]
        bsz = input.shape[1]

        if not (self.variant_flags_dict['include_globalnode_input']):
            pass  # not using the graph, embeddings only
        else:
            # Input signal from the graph:
            current_global_indices_ls = []
            graphs_in_batch_ls = []
            current_location_in_batchX_ls = []
            rows_to_skip = 0
            for t in range(seq_len):
                for currentglobal_vocab_idx in input[t, :]:
                    current_global_indices_ls.append(currentglobal_vocab_idx)
                    (x_indices_g, edge_index_g, _edge_type_g) = \
                        get_node_data(self.grapharea_matrix, currentglobal_vocab_idx, self.grapharea_size)
                    x_indices_g, edge_index_g = lemmatize_node(x_indices_g, edge_index_g, self)
                    sample_x = self.X.index_select(dim=0, index=x_indices_g.squeeze())
                    sample_graph = Data(x=sample_x, edge_index=edge_index_g)
                    graphs_in_batch_ls.append(sample_graph)

                    currentword_location_in_batchX = rows_to_skip + current_location_in_batchX_ls[-1] \
                        if len(current_location_in_batchX_ls) > 0 else 0
                    rows_to_skip = sample_x.shape[0]
                    current_location_in_batchX_ls.append(currentword_location_in_batchX)

            batch_graph = Batch.from_data_list(graphs_in_batch_ls)
            x_attention_states = self.gat_globals(batch_graph.x, batch_graph.edge_index)
            currentglobal_node_states = x_attention_states.index_select(dim=0, index=torch.tensor(
                current_location_in_batchX_ls).to(torch.int64).to(CURRENT_DEVICE))
            current_global_indices = torch.tensor(current_global_indices_ls).to(torch.int64).to(CURRENT_DEVICE)
            currentglobal_FT_embeds = (self.encoder.weight.data[:,0:self.d_inp]).index_select(dim=0, index=current_global_indices)
            currentglobal_nodestate_update = torch.cat([currentglobal_node_states, currentglobal_FT_embeds], dim=1)
            mask_for_scatter = make_2D_mask(current_global_indices, self.ntokens_vocab, self.d_encoding)
            self.encoder.weight.data.masked_scatter_(mask=mask_for_scatter, source=currentglobal_nodestate_update.clone())


        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):

            rnn.module.flatten_parameters()  # * now working
            current_input = raw_output
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
        encoding_size = self.d_inp * (1 + sum([int(self.variant_flags_dict[flag]) for flag in self.variant_flags_dict]))
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (encoding_size if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (encoding_size if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (encoding_size if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]



# the original AWD-LSTM is unchanged
class AWD(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers,
                 dropout=0.5, dropouth=0.5, dropouti=0.5,
                 dropoute=0.1, wdrop=0, tie_weights=False):
        super(AWD, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        # elif rnn_type == 'QRNN':
        #     from torchqrnn import QRNNLayer
        #     self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
        #     for rnn in self.rnns:
        #         rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights:
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
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            rnn.module.flatten_parameters()  # now working
            current_input = raw_output
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