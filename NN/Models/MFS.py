import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Graph.Adjacencies as AD
from NN.Models.Common import predict_globals_withGRU, init_model_parameters, init_common_architecture, get_input_signals
from NN.Models.RNNSteps import rnn_loop, reshape_tensor
from Utils import DEVICE
from torch.nn.parameter import Parameter
import logging
from time import time
import NN.ExplorePredictions as EP
import Utils
import nltk
from GetKBInputData.LemmatizeNyms import lemmatize_term
from enum import Enum
from NN.Models.SelectK import get_senseneighbours_of_k_globals
import pandas as pd
import os
import numpy as np
import Filesystem as F

class MFS(torch.nn.Module):

    def __init__(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 include_globalnode_input, batch_size, n_layers, n_hid_units, K, mfs_df):

        # -------------------- Initialization in common: parameters & globals --------------------
        super(MFS, self).__init__()
        init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                              include_globalnode_input,
                              batch_size, n_layers, n_hid_units)
        init_common_architecture(self, embeddings_matrix, graph_dataobj)

        # -------------------- Senses' architecture --------------------
        self.K = K
        self.grapharea_matrix_neighbours_section = self.grapharea_matrix[:, 0:self.grapharea_size]
        self.mfs_df = mfs_df

    # ---------------------------------------- Forward call ----------------------------------------

    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        # -------------------- Init --------------------
        distributed_batch_size = batchinput_tensor.shape[0]
        if not (distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            self.memory_hn = reshape_tensor(self.memory_hn, (self.n_layers, distributed_batch_size, self.hidden_size))
            self.hidden_state_bsize_adjusted = True

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

        # -------------------- Collect input signals
        word_embeddings = torch.stack(word_embeddings_ls, dim=0) if self.include_globalnode_input < 2 else None
        global_nodestates = torch.stack(currentglobal_nodestates_ls,
                                        dim=0) if self.include_globalnode_input > 0 else None
        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # ------------------- Globals ------------------
        seq_len = batch_input_signals.shape[0]
        predictions_globals, logits_globals = predict_globals_withGRU(self, batch_input_signals, seq_len, distributed_batch_size)

        # ------------------- Senses : pick MFS of the K=1 candidate -------------------
        if self.predict_senses:

            # ----- Find the k most likely globals, and retrieve their m.f.s -----
            k_globals_indices = logits_globals.sort(descending=True).indices[:, 0:1] # self.K==1 here
            sample_k_indices_in_vocab_lls = k_globals_indices.tolist()

            words_mfs_ls = []
            for s in range(distributed_batch_size * seq_len):
                k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
                k_globals_lemmatized = [self.vocabulary_lemmatizedList[idx] for idx in k_globals_vocab_indices] # 1 element
                lemmatized_word = k_globals_lemmatized[0]
                lemmatized_word_idx = Utils.word_to_vocab_index(lemmatized_word, self.vocabulary_wordList)
                mfs_word_row = self.mfs_df.loc[self.mfs_df[Utils.WORD + Utils.INDEX] == lemmatized_word_idx]
                try:
                    word_mfs_idx = mfs_word_row[Utils.MOST_FREQUENT_SENSE+Utils.INDEX].item()
                except ValueError: # every token has a sense, but (not in the text corpus) --> (not among the MFS)
                    sample_k_indices_lemmatized_ls = self.vocabulary_df[
                        self.vocabulary_df['word'].isin(k_globals_lemmatized)].index.to_list()
                    sample_k_indices_lemmatized = torch.tensor(sample_k_indices_lemmatized_ls).to(
                        CURRENT_DEVICE).squeeze(dim=0) + self.last_idx_senses
                    sense_neighbours_ls = get_senseneighbours_of_k_globals(model=self, sample_k_indices=sample_k_indices_lemmatized)
                    word_mfs_idx = sense_neighbours_ls[0].item() # since it is a nparray
                words_mfs_ls.append(word_mfs_idx)

            # ----- preparing the base for the artificial softmax -----
            senses_softmax = torch.ones((seq_len, distributed_batch_size, self.last_idx_senses)).to(CURRENT_DEVICE)
            epsilon = 10 ** (-8)
            senses_softmax = epsilon * senses_softmax  # base probability value for non-selected senses
            senses_mask = torch.zeros(size=(seq_len,distributed_batch_size, self.last_idx_senses)).to(torch.bool).to(CURRENT_DEVICE)

            words_mfs = torch.tensor(words_mfs_ls).reshape((seq_len, distributed_batch_size)).to(torch.torch.int64).to(CURRENT_DEVICE)
            # ----- writing in the artificial softmax -----
            for t in (range(seq_len)):
                for b in range(distributed_batch_size):
                    senses_mask[t, b, words_mfs[t,b]] = True
            assign_one = (torch.ones(
                size=senses_mask[senses_mask == True].shape)).to(CURRENT_DEVICE)
            senses_softmax.masked_scatter_(mask=senses_mask.data.clone(), source=assign_one)

            predictions_senses = torch.log(senses_softmax).reshape(seq_len * distributed_batch_size,
                                                                   senses_softmax.shape[2])

        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
