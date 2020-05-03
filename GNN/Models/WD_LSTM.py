import torch
import torch.nn.functional as tfunc
from time import time
from Utils import DEVICE, MAX_EDGES_PACKED
from torch.nn.parameter import Parameter
import GNN.Models.Common as C
from torchnlp.nn import WeightDropLSTM, WeightDrop
from torch.nn import LSTMCell

class WD_LSTM(torch.nn.Module):
    def __init__(self, data, grapharea_size, include_senses, n_layers, n_units):
        super(WD_LSTM, self).__init__()
        self.n_layers = n_layers
        self.include_senses = include_senses
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]

        # The embeddings matrix for: senses, globals, definitions, examples (the latter 2 may have gradient set to 0)
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)
        #self.select_first_node = Parameter(torch.tensor([0]).to(torch.float32), requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.d)), requires_grad=False)

        # Input signals: current global’s word embedding || global’s node-state (|| sense’s node state)
        self.concatenated_input_dim = self.d if not (self.include_senses) else 2 * self.d

        self.memory_hn_ls = [Parameter(torch.zeros(size=(1, n_units)), requires_grad=False)
                             for i in range(n_layers)]
        self.memory_cn_ls = [Parameter(torch.zeros(size=(1, n_units)), requires_grad=False) # self.d if i==0 else
                             for i in range(n_layers)]


        #self.wd_lstm = WeightDropLSTM(input_size=self.concatenated_input_dim, num_layers=n_layers, hidden_size=n_units)
        # we must use manual WeightDrop on LSTM cells, WeightDropLSTM is incompatible with PyTorch 1.4.0
        self.lstm_layers_ls = torch.nn.ModuleList(
                [LSTMCell(input_size=self.d, hidden_size=n_units)] +
                [LSTMCell(input_size=n_units, hidden_size=n_units) for _i in range(n_layers-1)])

        # 2nd part of the network as before: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=n_units,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)

        if self.include_senses:
            self.linear2sense = torch.nn.Linear(in_features=n_units,
                                                out_features=self.last_idx_senses, bias=True)


    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0

        predictions_globals_ls = []
        predictions_senses_ls = []
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        for i in range(self.n_layers):
            self.memory_hn_ls[i].detach_()
            self.memory_cn_ls[i].detach_()

        if batchinput_tensor.shape[0] > 1:
            sequences_in_the_batch_ls = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[0], dim=0)
        else:
            sequences_in_the_batch_ls = [batchinput_tensor]

        for padded_sequence in sequences_in_the_batch_ls:
            padded_sequence = padded_sequence.squeeze()
            padded_sequence = padded_sequence.chunk(chunks=padded_sequence.shape[0], dim=0)
            sequence_lts = [C.unpack_input_tensor(sample_tensor, self.N) for sample_tensor in padded_sequence]

            for ((x_indices_g, edge_index_g, edge_type_g), (x_indices_s, edge_index_s, edge_type_s)) in sequence_lts:
                # Input signal n.1: the current (global) word
                currentword_embedding = self.X.index_select(dim=0, index=x_indices_g[0])

                # Input signal n.2: the embedding of the current sense; + concatenating the input signals
                if self.include_senses:
                    if x_indices_s.nonzero().shape[0] == 0: # no sense was specified
                        currentsense_embedding = self.embedding_zeros
                    else: # sense was specified
                        currentsense_embedding = self.X.index_select(dim=0, index=x_indices_s[0])
                    input_signals = torch.cat([currentword_embedding, currentsense_embedding], dim=1)
                else:
                    input_signals = currentword_embedding

                #lstm_out, (hidden_n, cells_n) = self.wd_lstm(input_signals, (self.hidden_n, self.cells_n))
                # output of each layer:
                layer_out_ls = []
                # 1st layer, from input to hidden size
                (hidden_0, cells_0) = self.lstm_layers_ls[0](input_signals, (self.memory_hn_ls[0], self.memory_cn_ls[0]))
                layer_out_ls.append(hidden_0)
                self.memory_hn_ls[0].data.copy_(hidden_0.clone()) # store h in memory
                self.memory_cn_ls[0].data.copy_(cells_0.clone())  #
                # subsequent layers, from hidden size to hidden size
                for i in range(1, self.n_layers):
                    (hidden_i, cells_i) = self.lstm_layers_ls[i](layer_out_ls[i-1], (self.memory_hn_ls[i-1],
                                                                                     self.memory_cn_ls[i-1]))
                    layer_out_ls.append(hidden_i)
                    self.memory_hn_ls[i].data.copy_(hidden_i.clone())  # store h in memory
                    self.memory_cn_ls[i].data.copy_(cells_i.clone())   #

                lstm_out = layer_out_ls[-1]

                # 2nd part of the architecture: predictions
                logits_global = self.linear2global(lstm_out)  # shape=torch.Size([5])
                sample_predictions_globals = tfunc.log_softmax(logits_global, dim=1)
                predictions_globals_ls.append(sample_predictions_globals)
                if self.include_senses:
                    logits_sense = self.linear2sense(lstm_out)
                    sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
                    predictions_senses_ls.append(sample_predictions_senses)
                else:
                    predictions_senses_ls.append(torch.tensor(0).to(DEVICE)) # so I don't have to change the interface elsewhere

            #Utils.log_chronometer([t0,t1,t2,t3,t4,t5, t6, t7, t8])
        return torch.stack(predictions_globals_ls, dim=0).squeeze(), \
               torch.stack(predictions_senses_ls, dim=0).squeeze()
