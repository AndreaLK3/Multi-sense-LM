import torch
import Models.Variants.Common as Common
from Models.Variants.RNNSteps import rnn_loop, reshape_tensor
import logging
import Utils
from Models.Variants.SelectK import get_senseneighbours_of_k_globals
import numpy as np

class MFS(torch.nn.Module):

    def __init__(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 use_gold_lm, include_globalnode_input, batch_size, n_layers, n_hid_units, K, mfs_df):

        # -------------------- Initialization in common: parameters & globals --------------------
        super(MFS, self).__init__()
        Common.init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                              include_globalnode_input, use_gold_lm,
                              batch_size, n_layers, n_hid_units)
        Common.init_common_architecture(self, embeddings_matrix, graph_dataobj)

        # -------------------- Senses' architecture --------------------
        self.K = K
        self.grapharea_matrix_neighbours_section = self.grapharea_matrix[:, 0:self.grapharea_size]
        self.mfs_df = mfs_df

    # ---------------------------------------- Forward call ----------------------------------------

    def forward(self, batchinput_tensor, batch_labels):  # given the batches, the current node is at index 0
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
            Common.get_input_signals(self, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls)

        # -------------------- Collect input signals
        word_embeddings = torch.stack(word_embeddings_ls, dim=0) if self.include_globalnode_input < 2 else None
        global_nodestates = torch.stack(currentglobal_nodestates_ls,
                                        dim=0) if self.include_globalnode_input > 0 else None
        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # ------------------- Globals ------------------
        seq_len = batch_input_signals.shape[0]
        if not self.use_gold_lm:
            predictions_globals, logits_globals = Common.predict_globals_withGRU(self, batch_input_signals, seq_len, distributed_batch_size)
        else:
            predictions_globals = Common.assign_one(batch_labels[:,0], seq_len, distributed_batch_size,
                                            self.last_idx_globals - self.last_idx_senses, CURRENT_DEVICE)

        # ------------------- Senses : pick MFS of the K=1 candidate -------------------
        if self.predict_senses:

            # ----- Find the k most likely globals, and retrieve their m.f.s -----
            k_globals_indices = predictions_globals.sort(descending=True).indices[:, 0:1] # self.K==1 here
            sample_k_indices_in_vocab_lls = k_globals_indices.tolist()

            words_mfs_ls = []
            for s in range(distributed_batch_size * seq_len):
                k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
                k_globals_lemmatized = [self.vocabulary_lemmatizedList[idx] for idx in k_globals_vocab_indices] # 1 element
                lemmatized_word = k_globals_lemmatized[0]
                mfs_word_row = self.mfs_df.loc[self.mfs_df[Utils.WORD] == lemmatized_word]
                try:
                    word_mfs_idx = mfs_word_row["MostFrequentSenseindex"].values[0]
                except IndexError: # dummySenses do not appear in the MFS table, and must be retrieved separately
                    logging.debug("Did not find MFS for lemmatized_word=" + lemmatized_word)
                    lemmatized_word_idx = self.last_idx_senses + self.vocabulary_wordList.index(lemmatized_word)
                    senseneighbours_t = get_senseneighbours_of_k_globals(model=self, sample_k_indices=torch.tensor([lemmatized_word_idx]).to(torch.int64))
                    word_mfs_idx = senseneighbours_t.tolist()[0][0]
                words_mfs_ls.append(word_mfs_idx)

            words_mfs_t = torch.tensor(words_mfs_ls).reshape((seq_len, distributed_batch_size)).to(torch.torch.int64).to(CURRENT_DEVICE)
            predictions_senses = Common.assign_one(words_mfs_t, seq_len, distributed_batch_size, self.last_idx_senses, CURRENT_DEVICE)

        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
