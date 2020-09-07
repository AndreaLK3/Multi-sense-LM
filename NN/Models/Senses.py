import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Graph.Adjacencies as AD
from NN.Models.Common import unpack_input_tensor, init_model_parameters, lemmatize_node
from NN.Models.Steps_RNN import rnn_loop, reshape_memories
from Utils import DEVICE
from torch.nn.parameter import Parameter
import logging
import NN.ExplorePredictions as EP
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


def run_graphnet(t_input_lts, batch_elems_at_t, t_globals_indices_ls, CURRENT_DEVICE, model):

    currentglobal_nodestates_ls = []
    if model.include_globalnode_input:
        t_edgeindex_g_ls = [t_input_lts[b][0][1] for b in range(len(t_input_lts))]
        t_edgetype_g_ls = [t_input_lts[b][0][2] for b in range(len(t_input_lts))]
        for i_sample in range(batch_elems_at_t.shape[0]):
            sample_edge_index = t_edgeindex_g_ls[i_sample]
            sample_edge_type = t_edgetype_g_ls[i_sample]
            x_indices, edge_index, edge_type = lemmatize_node(t_globals_indices_ls[i_sample], sample_edge_index, sample_edge_type, model=model)
            sample_x = model.X.index_select(dim=0, index=x_indices.squeeze())
            x_attention_states = model.gat_globals(sample_x, edge_index)
            currentglobal_node_state = x_attention_states.index_select(dim=0, index=model.select_first_indices[0].to(
                torch.int64))
            currentglobal_nodestates_ls.append(currentglobal_node_state)

        t_currentglobal_node_states = torch.stack(currentglobal_nodestates_ls, dim=0).squeeze(dim=1)
        return t_currentglobal_node_states


class SelectK(torch.nn.Module):

    def __init__(self, model_type, data, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 include_globalnode_input,
                 batch_size, n_layers, n_hid_units, dropout_p):

        # -------------------- Initialization and parameters --------------------
        super(SelectK, self).__init__()
        self.model_type = model_type  # can be "LSTM" or "GRU"
        init_model_parameters(self, data, grapharea_size, grapharea_matrix, vocabulary_df, include_globalnode_input,
                                   batch_size, n_layers, n_hid_units, dropout_p)


        self.E = Parameter(embeddings_matrix.clone().detach(), requires_grad=True) # The matrix of embeddings
        self.dim_embs = self.E.shape[1]
        if include_globalnode_input:
            self.X = Parameter(data.x.clone().detach(), requires_grad=True)  # The matrix of global-nodestates

        # -------------------- Utilities --------------------
        # utility tensors, used in index_select etc.
        self.select_first_indices = Parameter(torch.tensor(list(range(n_hid_units))).to(torch.float32),requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        # Memories of the hidden states; overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_cn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))), requires_grad=False)
        self.memory_cn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))), requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # -------------------- Input signals --------------------
        self.concatenated_input_dim = self.dim_embs + int(include_globalnode_input) * Utils.GRAPH_EMBEDDINGS_DIM
        # GAT for the node-states from the dictionary graph
        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=Utils.GRAPH_EMBEDDINGS_DIM, out_channels=int(Utils.GRAPH_EMBEDDINGS_DIM / 2),
                                       heads=2)  # , node_dim=1)
            lemmatize_term('init', self.lemmatizer)# to establish LazyCorpusLoader and prevent a multi-thread crash

        # -------------------- The networks --------------------
        self.main_rnn_ls = torch.nn.ModuleList(
            [getattr(torch.nn, self.model_type)(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=512 if i == n_layers - 1 else n_hid_units, num_layers=1) for
             i in range(n_layers)])

        self.senses_rnn_ls = torch.nn.ModuleList(
            [getattr(torch.nn, self.model_type)(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=512 if i == n_layers - 1 else n_hid_units, num_layers=1)
             for i in range(n_layers)])

        # 2nd part of the network: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=512,  #
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        self.linear2senses = torch.nn.Linear(in_features=512,  #
                                                 out_features=self.last_idx_senses, bias=True)

    # ---------------------------------------- Forward call ----------------------------------------

    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        # -------------------- Init --------------------
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
        currentglobals_nodestates_ls = []

        for batch_elements_at_t in time_instants:
            batch_elems_at_t = batch_elements_at_t.squeeze(dim=1)
            elems_at_t_ls = batch_elements_at_t.chunk(chunks=batch_elems_at_t.shape[0], dim=0)

            t_input_lts = [unpack_input_tensor(sample_tensor, self.grapharea_size) for sample_tensor in elems_at_t_ls]
            t_globals_indices_ls = [t_input_lts[b][0][0] for b in range(len(t_input_lts))]

            # -------------------- Input --------------------
            # Input signal n.1: the embedding of the current (global) word
            t_current_globals_indices_ls = [x_indices[0]-self.last_idx_senses for x_indices in t_globals_indices_ls]
            t_current_globals_indices = torch.stack(t_current_globals_indices_ls, dim=0)
            t_word_embeddings = self.E.index_select(dim=0, index=t_current_globals_indices)
            word_embeddings_ls.append(t_word_embeddings)
            # Input signal n.2: the node-state of the current global word - now with graph batching
            if self.include_globalnode_input:
                t_g_nodestates = run_graphnet(t_input_lts, batch_elems_at_t,t_globals_indices_ls, CURRENT_DEVICE, self)
                currentglobals_nodestates_ls.append(t_g_nodestates)

        word_embeddings = torch.stack(word_embeddings_ls, dim=0)
        global_nodestates = torch.stack(currentglobals_nodestates_ls, dim=0) if self.include_globalnode_input else None

        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))  # , currentsense_node_state]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # ------------------- Globals -------------------
        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        task_1_out = rnn_loop(batch_input_signals, model=self, rnn_ls=self.main_rnn_ls)  # self.network_1_L1(input)
        task_1_out = task_1_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)

        seq_len = batch_input_signals.shape[0]
        task_1_out = task_1_out.reshape(distributed_batch_size * seq_len, task_1_out.shape[2])

        logits_global = self.linear2global(task_1_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)

        # ------------------- Senses -------------------
        # line 1: GRU for senses + linear FF-NN to logits.
        if self.predict_senses:
            task_2_out = rnn_loop(batch_input_signals, model=self, rnn_ls=self.senses_rnn_ls)
            task2_out = task_2_out.reshape(distributed_batch_size * seq_len, task_2_out.shape[2])

            logits_sense = self.linear2senses(task2_out)
            # predictions_senses = tfunc.log_softmax(logits_sense, dim=1)

            # line 2: select senses of the k most likely globals
            k_globals_indices = logits_global.sort(descending=True).indices[:, 0:self.k]

            senses_softmax = torch.ones((distributed_batch_size * seq_len, self.last_idx_senses)).to(CURRENT_DEVICE)
            epsilon = 10 ** (-8)
            senses_softmax = epsilon * senses_softmax  # base probability value for non-selected senses
            i_senseneighbours_mask = torch.zeros(size=(distributed_batch_size * seq_len, self.last_idx_senses)).to(torch.bool).to(CURRENT_DEVICE)

            sample_k_indices_in_vocab_lls = k_globals_indices.tolist()

            for s in range(distributed_batch_size * seq_len):
                k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
                k_globals_words = [self.vocabulary_wordlist[global_idx_in_vocab] for global_idx_in_vocab in
                                   k_globals_vocab_indices]
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
