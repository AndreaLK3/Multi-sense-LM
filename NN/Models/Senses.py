import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Graph.Adjacencies as AD
from NN.Models.Common import unpack_input_tensor, init_model_parameters, lemmatize_node, run_graphnet
from NN.Models.Steps_RNN import rnn_loop, reshape_memories
from Utils import DEVICE
from torch.nn.parameter import Parameter
import logging
from time import time
import NN.ExplorePredictions as EP
import Utils
import nltk
from PrepareKBInput.LemmatizeNyms import lemmatize_term
from enum import Enum

# ****** Auxiliary functions *******
def get_senseneighbours_of_k_globals(model, sample_k_indices):

    sample_neighbours_section = (model.grapharea_matrix[sample_k_indices.cpu(), 0:32].todense()) -1
    sample_neighbours = sample_neighbours_section[sample_neighbours_section > 0]
    sense_neighbours = sample_neighbours[sample_neighbours < model.last_idx_senses]
    return sense_neighbours

def subtract_probability_mass_from_selected(softmax_selected_senses, delta_to_subtract):
    max_index_t = torch.argmax(softmax_selected_senses)
    prev_max_value = softmax_selected_senses[max_index_t]
    softmax_selected_senses[max_index_t].data = prev_max_value - delta_to_subtract
    return softmax_selected_senses

# *****************************

# Choose among the senses of the most likely k=,1,5,10... globals.
# Add the [the probability distribution over those] to [the distribution over the whole senses' vocabulary]...

class SelectK(torch.nn.Module):

    def __init__(self, data, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 include_globalnode_input, batch_size, n_layers, n_hid_units, k):

        # -------------------- Initialization and parameters --------------------
        super(SelectK, self).__init__()
        init_model_parameters(self, data, grapharea_size, grapharea_matrix, vocabulary_df, include_globalnode_input,
                              batch_size, n_layers, n_hid_units)
        self.K = k

        self.E = Parameter(embeddings_matrix.clone().detach(), requires_grad=True)  # The matrix of embeddings
        self.dim_embs = self.E.shape[1]
        if include_globalnode_input:
            self.X = Parameter(data.x.clone().detach(), requires_grad=True)  # The matrix of global-nodestates

        # -------------------- Utilities --------------------
        # utility tensors, used in index_select etc.
        self.select_first_indices = Parameter(torch.tensor(list(range(2048))).to(torch.float32),requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        # Memories of the hidden states; overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))), requires_grad=False)

        self.hidden_state_bsize_adjusted = False

        # -------------------- Input signals --------------------
        self.concatenated_input_dim = self.dim_embs + int(include_globalnode_input) * Utils.GRAPH_EMBEDDINGS_DIM
        # GAT for the node-states from the dictionary graph
        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=Utils.GRAPH_EMBEDDINGS_DIM, out_channels=int(Utils.GRAPH_EMBEDDINGS_DIM / 2),
                                       heads=2)  # , node_dim=1)

        # -------------------- The networks --------------------
        self.main_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=512 if i == n_layers - 1 else n_hid_units, num_layers=1) for
             i in range(n_layers)])

        self.senses_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
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
        self.memory_hn_senses.detach_()

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
        task_1_out = rnn_loop(batch_input_signals, model=self, globals_or_senses_rnn=True)  # self.network_1_L1(input)
        task_1_out = task_1_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)

        seq_len = batch_input_signals.shape[0]
        task_1_out = task_1_out.reshape(distributed_batch_size * seq_len, task_1_out.shape[2])

        logits_global = self.linear2global(task_1_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)

        # ------------------- Senses -------------------
        # line 1: GRU for senses + linear FF-NN to logits.
        if self.predict_senses:
            task_2_out = rnn_loop(batch_input_signals, model=self, globals_or_senses_rnn=False)
            task2_out = task_2_out.reshape(distributed_batch_size * seq_len, task_2_out.shape[2])

            logits_sense = self.linear2senses(task2_out)
            # predictions_senses = tfunc.log_softmax(logits_sense, dim=1)

            # line 2: select senses of the k most likely globals
            k_globals_indices = logits_global.sort(descending=True).indices[:, 0:self.K]

            senses_softmax = torch.ones((distributed_batch_size * seq_len, self.last_idx_senses)).to(CURRENT_DEVICE)
            epsilon = 10 ** (-8)
            senses_softmax = epsilon * senses_softmax  # base probability value for non-selected senses
            i_senseneighbours_mask = torch.zeros(size=(distributed_batch_size * seq_len, self.last_idx_senses)).to(torch.bool).to(CURRENT_DEVICE)

            sample_k_indices_in_vocab_lls = k_globals_indices.tolist()

            for s in range(distributed_batch_size * seq_len):

                k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
                k_globals_lemmatized = [self.vocabulary_lemmatizedList[idx] for idx in k_globals_vocab_indices]
                sample_k_indices_lemmatized_ls = [
                    Utils.word_to_vocab_index(lemmatized_word, self.vocabulary_wordList) + self.last_idx_senses for
                    lemmatized_word in k_globals_lemmatized]
                sample_k_indices_lemmatized = torch.tensor(sample_k_indices_lemmatized_ls).to(CURRENT_DEVICE).squeeze(dim=0)

                sense_neighbours_t = torch.tensor(get_senseneighbours_of_k_globals(self, sample_k_indices_lemmatized))\
                    .squeeze(dim=0).to(torch.int64).to(CURRENT_DEVICE)

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

# ****** Auxiliary functions and elements *******

class QueryMethod(Enum):
    CONTEXT_AVERAGE = "average of the last C words of the context. C can be tuned"
    ATTENTION = "attention mechanism on the last C words of the context."
    OWN_GRU_OUT = "output of a separate GRU"
    GLOBALS_GRU = "output (or hidden layer, depending on the current version) of the globals' GRU"

def compute_query(method, model):
    if method == QueryMethod.CONTEXT_AVERAGE:
        pass
    else:
        raise Exception("QueryMethod not yet implemented.")

# *****************************


# To obtain a probability distribution over the senses of the most likely K globals, use a self-attention score:
# a query (e.g. the average of the last C words of the context, attention on the last words, or a state of the globals’ GRU, or the state of its own 1-layer GRU)
# and key vectors (the embeddings of the senses)
# => we compute 〖(q〗_c∙k_(s_i ))/√(d_k ) to get the self-attention logits, and the softmax to have a probability distribution over the senses of the selected K globals.
# We assign it, while keeping the other senses’ softmax at 10-8=~0, as usual
class SelfAttentionK(torch.nn.Module):

    def __init__(self, data, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 include_globalnode_input, batch_size, n_layers, n_hid_units, k):

        # -------------------- Initialization and parameters --------------------
        super(SelfAttentionK, self).__init__()
        init_model_parameters(self, data, grapharea_size, grapharea_matrix, vocabulary_df, include_globalnode_input,
                              batch_size, n_layers, n_hid_units)
        self.K = k

        self.E = Parameter(embeddings_matrix.clone().detach(), requires_grad=True)  # The matrix of embeddings
        self.dim_embs = self.E.shape[1]
        if include_globalnode_input:
            self.X = Parameter(data.x.clone().detach(), requires_grad=True)  # The matrix of global-nodestates

        # -------------------- Utilities --------------------
        # utility tensors, used in index_select etc.
        self.select_first_indices = Parameter(torch.tensor(list(range(2048))).to(torch.float32),requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        # Memories of the hidden states; overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))), requires_grad=False)

        self.hidden_state_bsize_adjusted = False

        # -------------------- Input signals --------------------
        self.concatenated_input_dim = self.dim_embs + int(include_globalnode_input) * Utils.GRAPH_EMBEDDINGS_DIM
        # GAT for the node-states from the dictionary graph
        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=Utils.GRAPH_EMBEDDINGS_DIM, out_channels=int(Utils.GRAPH_EMBEDDINGS_DIM / 2),
                                       heads=2)  # , node_dim=1)

        # -------------------- The networks --------------------
        self.main_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=512 if i == n_layers - 1 else n_hid_units, num_layers=1) for
             i in range(n_layers)])

        self.senses_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
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
        self.memory_hn_senses.detach_()

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
        task_1_out = rnn_loop(batch_input_signals, model=self, globals_or_senses_rnn=True)  # self.network_1_L1(input)
        task_1_out = task_1_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)

        seq_len = batch_input_signals.shape[0]
        task_1_out = task_1_out.reshape(distributed_batch_size * seq_len, task_1_out.shape[2])

        logits_global = self.linear2global(task_1_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)

        # ------------------- Senses -------------------
        # line 1: GRU for senses + linear FF-NN to logits.
        if self.predict_senses:
            task_2_out = rnn_loop(batch_input_signals, model=self, globals_or_senses_rnn=False)
            task2_out = task_2_out.reshape(distributed_batch_size * seq_len, task_2_out.shape[2])

            logits_sense = self.linear2senses(task2_out)
            # predictions_senses = tfunc.log_softmax(logits_sense, dim=1)

            # line 2: select senses of the k most likely globals
            k_globals_indices = logits_global.sort(descending=True).indices[:, 0:self.K]

            senses_softmax = torch.ones((distributed_batch_size * seq_len, self.last_idx_senses)).to(CURRENT_DEVICE)
            epsilon = 10 ** (-8)
            senses_softmax = epsilon * senses_softmax  # base probability value for non-selected senses
            i_senseneighbours_mask = torch.zeros(size=(distributed_batch_size * seq_len, self.last_idx_senses)).to(torch.bool).to(CURRENT_DEVICE)

            sample_k_indices_in_vocab_lls = k_globals_indices.tolist()

            for s in range(distributed_batch_size * seq_len):

                k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
                k_globals_lemmatized = [self.vocabulary_lemmatizedList[idx] for idx in k_globals_vocab_indices]
                sample_k_indices_lemmatized_ls = [
                    Utils.word_to_vocab_index(lemmatized_word, self.vocabulary_wordList) + self.last_idx_senses for
                    lemmatized_word in k_globals_lemmatized]
                sample_k_indices_lemmatized = torch.tensor(sample_k_indices_lemmatized_ls).to(CURRENT_DEVICE).squeeze(dim=0)

                sense_neighbours_t = torch.tensor(get_senseneighbours_of_k_globals(self, sample_k_indices_lemmatized))\
                    .squeeze(dim=0).to(torch.int64).to(CURRENT_DEVICE)

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