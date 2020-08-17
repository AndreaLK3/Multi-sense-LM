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
from GNN.Models.Steps_RNN import reshape_memories, select_layer_memory, update_layer_memory

class NN(torch.nn.Module):

    def __init__(self, model_type, data, grapharea_size, grapharea_matrix, vocabulary_df,
                 include_globalnode_input, include_sensenode_input, predict_senses,
                 batch_size, n_layers, n_hid_units, dropout_p):
        super(NN, self).__init__()
        self.model_type = model_type  # can be "LSTM" or "GRU"
        init_model_parameters(self, data, grapharea_size, grapharea_matrix, vocabulary_df,
                                   include_globalnode_input, include_sensenode_input, predict_senses,
                                   batch_size, n_layers, n_hid_units, dropout_p)
        self.num_embs = data.x.shape[0]
        self.dim_embs = data.x.shape[1]

        # the embeddings matrices, here we create 2 distinct ones at random
        range_random_embs = 10
        self.embs_A = (torch.rand((self.num_embs, self.dim_embs)) - 0.5) * range_random_embs
        self.embs_B = (torch.rand((self.num_embs, self.dim_embs)) - 0.5) * range_random_embs

        # utility tensor, used in index_select etc.
        self.select_first_indices = Parameter(torch.tensor(list(range(n_hid_units))).to(torch.float32),
                                              requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        self.concatenated_input_dim = self.dim_embs * (1 + int(include_globalnode_input) + int(include_sensenode_input))

        # Memories for the hidden states; overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_cn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))),
                                          requires_grad=False)
        self.memory_cn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))),
                                          requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # the Networks: we start by using FF-NN, then move on to GRUs
        #self.network_1_L1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=n_hid_units)
        self.main_rnn_ls = torch.nn.ModuleList(
            [getattr(torch.nn, self.model_type)(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=512 if i == n_layers - 1 else n_hid_units, num_layers=1) for
             i in range(n_layers)])

        if predict_senses:
            #self.network_2_L1 = torch.nn.Linear(in_features=self.concatenated_input_dim, out_features=n_hid_units)
            self.senses_rnn_ls = torch.nn.ModuleList(
                [getattr(torch.nn, self.model_type)(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                    hidden_size=512 if i == n_layers - 1 else n_hid_units, num_layers=1)
                 for i in range(n_layers)])

        # 2nd part of the network: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=512, #
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        if predict_senses:

            self.linear2senses = torch.nn.Linear(in_features=512, # 
                                                 out_features=self.last_idx_senses, bias=True)

    # ------------
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

        word_embeddings_ls_1 = []
        word_embeddings_ls_2 = []
        currentglobal_nodestates_ls = []

        for batch_elements_at_t in time_instants:
            batch_elems_at_t = batch_elements_at_t.squeeze(dim=1)
            elems_at_t_ls = batch_elements_at_t.chunk(chunks=batch_elems_at_t.shape[0], dim=0)

            t_input_lts = [unpack_input_tensor(sample_tensor, self.grapharea_size) for sample_tensor in elems_at_t_ls]

            t_globals_indices_ls = [t_input_lts[b][0][0] for b in range(len(t_input_lts))]

            # Input signal n.1: the embedding of the current (global) word
            t_current_globals_indices_ls = [x_indices[0] for x_indices in t_globals_indices_ls]
            t_current_globals_indices = torch.stack(t_current_globals_indices_ls, dim=0)
            t_word_embeddings_1 = self.embs_A.index_select(dim=0, index=t_current_globals_indices)
            t_word_embeddings_2 = self.embs_A.index_select(dim=0, index=t_current_globals_indices)
            word_embeddings_ls_1.append(t_word_embeddings_1)
            word_embeddings_ls_2.append(t_word_embeddings_2)


        word_embeddings_1 = torch.stack(word_embeddings_ls_1, dim=0)
        word_embeddings_2 = torch.stack(word_embeddings_ls_2, dim=0)
        global_nodestates = torch.stack(currentglobal_nodestates_ls, dim=0) if self.include_globalnode_input else None

        batch_input_signals_1_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings_1, global_nodestates]))  # , currentsense_node_state]))
        batch_input_signals_1 = torch.cat(batch_input_signals_1_ls, dim=2)

        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        # task_1_out = None
        task_1_out = rnn_loop(batch_input_signals_1, model=self)  # self.network_1_L1(input)

        task_1_out = task_1_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)
        seq_len = batch_input_signals_1.shape[0]
        task_1_out = task_1_out.reshape(distributed_batch_size * seq_len, task_1_out.shape[2])

        # 2nd part of the architecture: predictions
        # globals
        logits_global = self.linear2global(task_1_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)

        # senses
        # line 1: GRU for senses + linear FF-NN tos logits.
        if self.predict_senses:
            # task2_out = None
            batch_input_signals_2_ls = list(filter(lambda signal: signal is not None,
                                                   [word_embeddings_2,
                                                    global_nodestates]))  # , currentsense_node_state]))
            batch_input_signals_2 = torch.cat(batch_input_signals_2_ls, dim=2)

            task_2_out = rnn_loop(batch_input_signals_2, model=self)

            task2_out = task_2_out.reshape(distributed_batch_size * seq_len, task_2_out.shape[2])

            logits_sense = self.linear2senses(task2_out)

            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses

