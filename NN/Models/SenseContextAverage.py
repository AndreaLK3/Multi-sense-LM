import torch
from NN.Models.Common import predict_globals_withGRU, init_model_parameters, init_common_architecture, \
    get_input_signals
from NN.Models.RNNSteps import reshape_memories, reshape_tensor
from torch.nn.parameter import Parameter
import Utils
from NN.Models.SelectK import get_senseneighbours_of_k_globals
import numpy as np
import os
import Filesystem as F

# May move it / refactor
def update_context_average(location_context, word_embeddings, prev_word_embeddings, num_C, CURRENT_DEVICE):

    current_and_prev_word_embeddings = torch.cat([prev_word_embeddings, word_embeddings], dim=0)
    # logging.info("current_and_prev_word_embeddings=" + str(current_and_prev_word_embeddings))
    loc_ctx_toadd = torch.zeros(size=location_context.shape).to(CURRENT_DEVICE)
    # for every time instant t, we update the location context:
    T = word_embeddings.shape[0]
    for t in (range(0, T)):
        loc_ctx_toadd[t, : ,:] = torch.sum(current_and_prev_word_embeddings[T+t-num_C+1 : T+t+1, :,:], dim=0)
        # logging.info("t=" + str(t) + " ; loc_ctx_toadd.data / num_C = " + str(loc_ctx_toadd.data / num_C))
    location_context.data = location_context.data + loc_ctx_toadd.data / num_C
    return location_context


class SenseContextAverage(torch.nn.Module):

    def __init__(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 include_globalnode_input, batch_size, n_layers, n_hid_units, K, num_C):

        # -------------------- Initialization in common: parameters & globals --------------------
        super(SenseContextAverage, self).__init__()

        init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                              include_globalnode_input,
                              batch_size, n_layers, n_hid_units)
        init_common_architecture(self, embeddings_matrix, graph_dataobj)

        # -------------------- Senses' architecture --------------------

        self.K = K
        self.grapharea_matrix_neighbours_section = self.grapharea_matrix[:, 0:self.grapharea_size]
        self.num_C = num_C

        precomputed_SC_filepath = os.path.join(F.FOLDER_INPUT, F.FOLDER_SENSELABELED, str(num_C) + F.MATRIX_SENSE_CONTEXTS_FILEEND)
        senses_average_context_SC = np.load(precomputed_SC_filepath)
        self.SC = Parameter(torch.tensor(senses_average_context_SC), requires_grad=False)
        self.prev_word_embeddings = Parameter(torch.zeros((200, batch_size, self.E.shape[1])), requires_grad=False)
        # prev_word_embeddings will be reshaped when we know the seq_len and distributed_batch_size

        self.location_context = Parameter(torch.zeros(size=(200, batch_size, self.E.shape[1])), requires_grad=False)
        # location_context will be reshape
        self.ctx_tensors_adjusted = False
        self.cosine_sim = torch.nn.CosineSimilarity(dim=3)

    # ---------------------------------------- Forward call ----------------------------------------

    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        # -------------------- Init --------------------
        distributed_batch_size = batchinput_tensor.shape[0]
        if not (distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            self.memory_hn = reshape_tensor(self.memory_hn, (self.n_layers, distributed_batch_size, self.hidden_size))
            self.hidden_state_bsize_adjusted = True
        if not(self.ctx_tensors_adjusted):
            self.location_context = reshape_tensor(self.location_context, (batchinput_tensor.shape[1], distributed_batch_size, self.E.shape[1]))
            self.prev_word_embeddings = reshape_tensor(self.prev_word_embeddings, (batchinput_tensor.shape[1], distributed_batch_size, self.E.shape[1]))
            self.ctx_tensors_adjusted = True

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

            all_sense_neighbours_ls = []
            sense_neighbours_len_ls = []
            for s in range(distributed_batch_size * seq_len):

                k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
                k_globals_lemmatized = [self.vocabulary_lemmatizedList[idx] for idx in k_globals_vocab_indices]

                sample_k_indices_lemmatized_ls = [
                    Utils.word_to_vocab_index(lemmatized_word, self.vocabulary_wordList)for
                    lemmatized_word in k_globals_lemmatized]
                sample_k_indices_lemmatized = torch.tensor(sample_k_indices_lemmatized_ls).to(CURRENT_DEVICE).squeeze(dim=0) + self.last_idx_senses

                sense_neighbours = torch.tensor(get_senseneighbours_of_k_globals(self, sample_k_indices_lemmatized))\
                    .squeeze(dim=0).to(torch.int64).to(CURRENT_DEVICE)
                sense_neighbours_len_ls.append(sense_neighbours.shape[0])

                random_sense_idx = np.random.randint(low=0, high=self.SC.shape[0]) # we use a random sense as pad/filler. In the case where it's actually closer than our selected senses, we will chose that one.
                all_sense_neighbours_ls.append(torch.nn.functional.pad(input=sense_neighbours, pad=[0, self.grapharea_size - sense_neighbours.shape[0]], value=random_sense_idx))


            # ------------------- Senses: compare location context with sense average context -------------------
            # prepare the base for the artificial softmax
            senses_softmax = torch.ones((seq_len, distributed_batch_size, self.last_idx_senses)).to(CURRENT_DEVICE)
            epsilon = 10 ** (-8)
            senses_softmax = epsilon * senses_softmax  # base probability value for non-selected senses
            senses_mask = torch.zeros(size=(seq_len,distributed_batch_size, self.last_idx_senses)).to(torch.bool).to(CURRENT_DEVICE)
            quantity_to_subtract_from_selected = epsilon * (self.last_idx_senses - 1)
            quantity_to_assign_to_chosen_sense = 1 - quantity_to_subtract_from_selected

            # update the location context with the latest word embeddings
            update_context_average(self.location_context, word_embeddings, self.prev_word_embeddings, self.num_C, CURRENT_DEVICE)
            self.prev_word_embeddings.data = word_embeddings.data

            # the context of the selected senses and the cosine similarity:
            senses_context = torch.zeros((self.location_context.shape[0], self.location_context.shape[1],
                                         self.grapharea_size * self.K, self.location_context.shape[2]))

            all_sense_neighbours = torch.stack(all_sense_neighbours_ls, dim=0).reshape((seq_len, distributed_batch_size, self.grapharea_size))
            senses_context.data = self.SC[all_sense_neighbours,:].data
            samples_cosinesim = self.cosine_sim(self.location_context.unsqueeze(2), senses_context)
            samples_sortedindices = torch.sort(samples_cosinesim, descending=True).indices
            samples_firstindex = samples_sortedindices[:,:,0]
            samples_firstsense = torch.gather(all_sense_neighbours, dim=2, index=samples_firstindex.unsqueeze(2))

            # writing in the artificial softmax

            for t in (range(seq_len)):
                for b in range(distributed_batch_size):
                    senses_mask[t, b, samples_firstsense[t,b].item()] = True
            assign_one = (torch.ones(size=senses_mask[senses_mask==True].shape) * quantity_to_assign_to_chosen_sense).to(CURRENT_DEVICE)
            senses_softmax.masked_scatter_(mask=senses_mask.data.clone(), source=assign_one)

            predictions_senses = torch.log(senses_softmax).reshape(seq_len * distributed_batch_size, senses_softmax.shape[2])

        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
