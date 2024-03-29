import torch

import Lexicon
import Models.Variants.Common as Common
import Models.Variants.InputSignals
from Models.Variants.RNNSteps import rnn_loop, reshape_tensor
import logging
import Utils
from Models.Variants.SelectK import get_senseneighbours_of_k_globals
import numpy as np

class MFS(torch.nn.Module):

    def __init__(self, standardLM, graph_dataobj, grapharea_size, grapharea_matrix,
                 vocabulary_df, batch_size, n_layers, n_hid_units, K, mfs_df):

        super(MFS, self).__init__()

        self.StandardLM = standardLM
        Common.init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                                     batch_size, n_layers, n_hid_units)

        # -------------------- Senses' architecture --------------------
        self.K = K
        self.grapharea_matrix_neighbours_section = self.grapharea_matrix[:, 0:self.grapharea_size]
        self.mfs_df = mfs_df

    # ---------------------------------------- Forward call ----------------------------------------

    def forward(self, batchinput_tensor, batch_labels):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        batch_input_signals, globals_input_ids_ls, word_embeddings, predictions_globals = \
            Common.get_input_and_predict_globals(self, batchinput_tensor, batch_labels)

        # ------------------- Senses : pick MFS of the K=1 candidate -------------------
        seq_len = batch_input_signals.shape[0]
        if self.predict_senses:
            # ----- Find the k most likely globals, and retrieve their m.f.s -----
            k_globals_indices = predictions_globals.sort(descending=True).indices[:, 0:1] # self.K==1 here
            sample_k_indices_in_vocab_lls = k_globals_indices.tolist()

            words_mfs_ls = []
            for s in range(self.batch_size * seq_len):
                k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
                k_globals_lemmatized = [self.vocabulary_lemmatizedList[idx] for idx in k_globals_vocab_indices] # 1 element
                lemmatized_word = k_globals_lemmatized[0]
                mfs_word_row = self.mfs_df.loc[self.mfs_df[Lexicon.WORD] == lemmatized_word]
                try:
                    word_mfs_idx = mfs_word_row["MostFrequentSenseindex"].values[0]
                except IndexError: # dummySenses do not appear in the MFS table, and must be retrieved separately
                    logging.debug("Did not find MFS for lemmatized_word=" + lemmatized_word)
                    lemmatized_word_idx = self.last_idx_senses + self.vocabulary_wordList.index(lemmatized_word)
                    senseneighbours_t = get_senseneighbours_of_k_globals(model=self, sample_k_indices=torch.tensor([lemmatized_word_idx]).to(torch.int64))
                    word_mfs_idx = senseneighbours_t.tolist()[0][0]
                words_mfs_ls.append(word_mfs_idx)

            words_mfs_t = torch.tensor(words_mfs_ls).reshape((seq_len, self.batch_size)).to(torch.torch.int64).to(CURRENT_DEVICE)
            predictions_senses = Common.assign_one(words_mfs_t, seq_len, self.batch_size, self.last_idx_senses, CURRENT_DEVICE)

        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
