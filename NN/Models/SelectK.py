import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Graph.Adjacencies as AD
from NN.Models.Common import predict_globals_withGRU, init_model_parameters, init_common_architecture, get_input_signals
from NN.Models.RNNSteps import rnn_loop, reshape_memories
from Utils import DEVICE
from torch.nn.parameter import Parameter
import logging
from time import time
import NN.ExplorePredictions as EP
import Utils
import nltk
from GetKBInputData.LemmatizeNyms import lemmatize_term
from enum import Enum

# ****** Auxiliary functions *******
def get_senseneighbours_of_k_globals(model, sample_k_indices):

    sample_neighbours_section = (model.grapharea_matrix[sample_k_indices.cpu(), 0:32].todense()) -1
    sample_neighbours = sample_neighbours_section[sample_neighbours_section >= 0]
    sense_neighbours = sample_neighbours[sample_neighbours < model.last_idx_senses]
    return sense_neighbours

def get_senseneighbours_of_k_globals_3D(model, graphmat_neighbours_section, k_indices):

    k_indices_flat = k_indices.flatten()
    neighbours_rows = graphmat_neighbours_section.index_select(dim=0, index=k_indices_flat)

    grouped_neighbors_rows_ls = [neighbours_rows[i:i + model.K] for i in range(0, neighbours_rows.shape[0], model.K)]
    grouped_neighbours_rows = torch.stack(grouped_neighbors_rows_ls, dim=0) # it should be of torch.Size([(bsz * seq_len), K, grapharea_size])
    grouped_neighbours_rows_adjusted = grouped_neighbours_rows -1


    return grouped_neighbours_rows

def subtract_probability_mass_from_selected(softmax_selected_senses, delta_to_subtract):
    max_index_t = torch.argmax(softmax_selected_senses)
    prev_max_value = softmax_selected_senses[max_index_t]
    softmax_selected_senses[max_index_t].data = prev_max_value - delta_to_subtract
    return softmax_selected_senses

# *****************************

# Choose among the senses of the most likely k=,1,5,10... globals.
# Add the [the probability distribution over those] to [the distribution over the whole senses' vocabulary]...

class SelectK(torch.nn.Module):

    def __init__(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 include_globalnode_input, batch_size, n_layers, n_hid_units, k):

        # -------------------- Initialization in common: parameters & globals --------------------
        super(SelectK, self).__init__()

        init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                              include_globalnode_input,
                              batch_size, n_layers, n_hid_units)
        init_common_architecture(self, embeddings_matrix, graph_dataobj)

        # -------------------- Senses' architecture --------------------

        self.K = k
        self.grapharea_matrix_neighbours_section = self.grapharea_matrix[:, 0:self.grapharea_size]


        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))),
                                          requires_grad=False)
        self.senses_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                          hidden_size=n_hid_units // 2 if i == n_layers - 1 else n_hid_units, num_layers=1)  # 512
             for i in range(n_layers)])

        self.linear2senses = torch.nn.Linear(in_features=n_hid_units // 2,  # 512
                                             out_features=self.last_idx_senses, bias=True)


    # ---------------------------------------- Forward call ----------------------------------------

    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        # -------------------- Init --------------------
        t0 = time()
        distributed_batch_size = batchinput_tensor.shape[0]
        if not (distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            reshape_memories(distributed_batch_size, self)
            # hidden_state_bsize_adjusted is set to True in reshape_memories

        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()
        self.memory_hn_senses.detach_()

        if batchinput_tensor.shape[1] > 1:
            time_instants = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[1], dim=1)
        else:
            time_instants = [batchinput_tensor]

        word_embeddings_ls = []
        currentglobal_nodestates_ls = []
        # -------------------- Compute input signals -------------------
        for batch_elements_at_t in time_instants:
            get_input_signals(self, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls)

        # -------------------- Collect input signals
        word_embeddings = torch.stack(word_embeddings_ls, dim=0)
        global_nodestates = torch.stack(currentglobal_nodestates_ls, dim=0) if self.include_globalnode_input else None
        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # ------------------- Globals ------------------
        seq_len = batch_input_signals.shape[0]
        predictions_globals, logits_globals = predict_globals_withGRU(self, batch_input_signals, seq_len, distributed_batch_size)

        # ------------------- Senses -------------------
        # line 1: GRU for senses + linear FF-NN to logits.
        if self.predict_senses:
            task_2_out = rnn_loop(batch_input_signals, model=self, globals_or_senses_rnn=False)
            task2_out = task_2_out.reshape(distributed_batch_size * seq_len, task_2_out.shape[2])
            logits_senses = self.linear2senses(task2_out)

            # line 2: select senses of the k most likely globals
            k_globals_indices = logits_globals.sort(descending=True).indices[:, 0:self.K]

            senses_softmax = torch.ones((distributed_batch_size * seq_len, self.last_idx_senses)).to(CURRENT_DEVICE)
            epsilon = 10 ** (-8)
            senses_softmax = epsilon * senses_softmax  # base probability value for non-selected senses
            i_senseneighbours_mask = torch.zeros(size=(distributed_batch_size * seq_len, self.last_idx_senses)).to(torch.bool).to(CURRENT_DEVICE)

            sample_k_indices_in_vocab_lls = k_globals_indices.tolist()

            # New method
            lemmatized_k_indices_ls = []

            # for s in range(distributed_batch_size * seq_len):
            #     k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
            #     k_globals_lemmatized = [self.vocabulary_lemmatizedList[idx] for idx in k_globals_vocab_indices]
            #     sample_k_indices_lemmatized_ls = [
            #         Utils.word_to_vocab_index(lemmatized_word, self.vocabulary_wordList) + self.last_idx_senses for
            #         lemmatized_word in k_globals_lemmatized]
            #     sample_k_indices_lemmatized = torch.tensor(sample_k_indices_lemmatized_ls).to(CURRENT_DEVICE).squeeze(dim=0)
            #     lemmatized_k_indices_ls.append(sample_k_indices_lemmatized)
            # lemmatized_k_indices = torch.stack(lemmatized_k_indices_ls, dim=0)
            #
            # sense_neighbours_t = torch.tensor(get_senseneighbours_of_k_globals(self, sample_k_indices_lemmatized)) \
            #     .squeeze(dim=0).to(torch.int64).to(CURRENT_DEVICE)
            # t2 = time()
            #
            # # standard procedure: get the logits of the senses of the most likely globals,
            # # apply a softmax only over them, and then assign an epsilon probability to the other senses
            # sample_logits_senses = logits_senses.index_select(dim=0, index=self.select_first_indices[s].to(
            #     torch.int64)).squeeze()
            # logits_selected_senses = sample_logits_senses.index_select(dim=0, index=sense_neighbours_t)
            # softmax_selected_senses = tfunc.softmax(input=logits_selected_senses, dim=0)

            for s in range(distributed_batch_size * seq_len):
                t0=time()
                k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
                k_globals_lemmatized = [self.vocabulary_lemmatizedList[idx] for idx in k_globals_vocab_indices]
                t1 = time()
                sample_k_indices_lemmatized_ls = [
                    Utils.word_to_vocab_index(lemmatized_word, self.vocabulary_wordList)for
                    lemmatized_word in k_globals_lemmatized]
                sample_k_indices_lemmatized = torch.tensor(sample_k_indices_lemmatized_ls).to(CURRENT_DEVICE).squeeze(dim=0) + self.last_idx_senses
                t2 = time()

                sense_neighbours_t = torch.tensor(get_senseneighbours_of_k_globals(self, sample_k_indices_lemmatized))\
                    .squeeze(dim=0).to(torch.int64).to(CURRENT_DEVICE)
                t3 = time()

                # standard procedure: get the logits of the senses of the most likely globals,
                # apply a softmax only over them, and then assign an epsilon probability to the other senses
                sample_logits_senses = logits_senses.index_select(dim=0, index=self.select_first_indices[s].to(torch.int64)).squeeze()
                logits_selected_senses = sample_logits_senses.index_select(dim=0, index=sense_neighbours_t)
                softmax_selected_senses = tfunc.softmax(input=logits_selected_senses, dim=0)
                t4 = time()

                # for i in range(len(sense_neighbours_t)):
                #     i_senseneighbours_mask[s,sense_neighbours_t[i]]=True
                i_senseneighbours_mask[s, sense_neighbours_t] = True
                t5 = time()
                quantity_to_subtract_from_selected = epsilon * (self.last_idx_senses - len(sense_neighbours_t))
                softmax_selected_senses = subtract_probability_mass_from_selected(softmax_selected_senses, quantity_to_subtract_from_selected)
                senses_softmax[s].masked_scatter_(mask=i_senseneighbours_mask[s].data.clone(), source=softmax_selected_senses)
                #Utils.log_chronometer([t0, t1, t2, t3, t4, t5])
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
