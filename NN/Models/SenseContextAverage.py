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
from NN.Models.SelectK import get_senseneighbours_of_k_globals, subtract_probability_mass_from_selected
import numpy as np
import os
import Filesystem as F

class SenseContextAverage(torch.nn.Module):

    def __init__(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 include_globalnode_input, batch_size, n_layers, n_hid_units, k, num_C):

        # -------------------- Initialization in common: parameters & globals --------------------
        super(SenseContextAverage, self).__init__()

        init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                              include_globalnode_input,
                              batch_size, n_layers, n_hid_units)
        init_common_architecture(self, embeddings_matrix, graph_dataobj)

        # -------------------- Senses' architecture --------------------

        self.K = k
        self.grapharea_matrix_neighbours_section = self.grapharea_matrix[:, 0:self.grapharea_size]
        self.num_C = num_C

        precomputed_SC_filepath = os.path.join(F.FOLDER_INPUT, F.FOLDER_SENSELABELED, str(num_C) + F.MATRIX_SENSE_CONTEXTS_FILEEND)
        senses_average_context_SC = np.load(precomputed_SC_filepath)
        self.SC = Parameter(torch.tensor(senses_average_context_SC), requires_grad=False)

        self.location_contexts = Parameter(torch.tensor([])) # starts as a Parameter here to have several versions via DataParallel
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)

    # ---------------------------------------- Forward call ----------------------------------------

    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        # -------------------- Init --------------------
        t0 = time()
        distributed_batch_size = batchinput_tensor.shape[0]
        if not (distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            reshape_memories(distributed_batch_size, self) # # hidden_state_bsize_adjusted=True in reshape_memories
        if self.location_contexts.shape[0]==0:
            self.location_contexts = Parameter(torch.zeros(batchinput_tensor.shape[1],distributed_batch_size, self.E.shape[1]))

        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()

        if batchinput_tensor.shape[1] > 1:
            time_instants = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[1], dim=1)
        else:
            time_instants = [batchinput_tensor]

        word_embeddings_ls = []
        currentglobal_nodestates_ls = []
        # -------------------- Compute input signals -------------------
        for batch_elements_at_t in time_instants:
            get_input_signals(self, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls)

        # -------------------- Collect input signals -------------------
        word_embeddings = torch.stack(word_embeddings_ls, dim=0)
        global_nodestates = torch.stack(currentglobal_nodestates_ls, dim=0) if self.include_globalnode_input else None
        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # ------------------- Globals ------------------
        seq_len = batch_input_signals.shape[0]
        predictions_globals, logits_globals = predict_globals_withGRU(self, batch_input_signals, seq_len, distributed_batch_size)

        # ------------------- Senses -------------------
        if self.predict_senses:

            # Select the senses of the k most likely globals
            k_globals_indices = logits_globals.sort(descending=True).indices[:, 0:self.K]
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
            all_sense_neighbours_ls = 
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

                sense_neighbours = torch.tensor(get_senseneighbours_of_k_globals(self, sample_k_indices_lemmatized))\
                    .squeeze(dim=0).to(torch.int64).to(CURRENT_DEVICE)
                t3 = time()

                # ------------------- Senses: compare location context with sense average context -------------------

                #update the location context with the latest word embeddings
                self.location_contexts.data = (self.location_contexts.data + word_embeddings) / self.num_C


        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
