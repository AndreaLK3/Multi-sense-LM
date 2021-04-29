import torch.nn.functional as tfunc
from math import sqrt
import torch
from Models.Variants.Common import init_model_parameters, ContextMethod, assign_one
from Models.Variants.InputSignals import get_input_signals
from Models.Variants.RNNSteps import reshape_tensor, rnn_loop
from torch.nn.parameter import Parameter
import Utils
from Models.Variants.SelectK import get_senseneighbours_of_k_globals
import numpy as np
import os
import Filesystem as F
from Models.Variants.SenseContext import update_context_average, init_context_handling
import logging

###################################
### 0: Self-attention mechanism ###
###################################

class ComputeLogits(torch.nn.Module):
    # if operating with multiple heads, I use concatenation
    def __init__(self, dim_input_context, dim_input_elems, dim_qkv, k, grapharea_size):
        super(ComputeLogits, self).__init__()
        self.d_input_context = dim_input_context
        self.d_input_elems = dim_input_elems
        self.d_qkv = dim_qkv  # the dimensionality of queries, keys and values - down from self.d_input
        self.k = k
        self.grapharea_size = grapharea_size

        self.Wq = torch.nn.Linear(in_features=self.d_input_context,out_features=self.d_qkv, bias=False)
        self.Wk = torch.nn.Linear(in_features=self.d_input_elems, out_features=self.d_qkv, bias=False)
        # self.logitsMultiplier = Parameter(torch.tensor(1000.0), requires_grad=True)

    def forward(self, input_q, input_kv):

        # Self-attention:
        # query, obtained projecting the representation of the local context
        input_query = input_q
        query = self.Wq(input_query)

        # <= k keys, obtained projecting the embeddings of the selected senses
        keys = self.Wk(input_kv)

        # Formula for self-attention scores: softmax{(query*key)/sqrt(d_k)}
        qk = torch.matmul(query.unsqueeze(2), keys.permute(0, 1, 3, 2))

        # selfatt_logits_0 = qk.squeeze().view(keys.shape[0]*keys.shape[1], self.grapharea_size*self.k) # torch.matmul(query_2D, keys_2D.t()).diagonal().reshape(keys.shape[0], keys.shape[1], keys.shape[2])
        selfatt_logits_1 = qk.squeeze(dim=2) / sqrt(self.d_qkv)# torch.clamp(self.logitsMultiplier, min=10**(-6), max=1000) # numerical adjustment, otherwise Q*K is too small
        # and in that case any softmax gives a ~uniform distribution.

        # logging.info("selfatt_logits_1[0:3, 0:2, 0:5]=" + str(selfatt_logits_1[0:3, 0:2, 0:5]))
        return selfatt_logits_1

# To obtain a probability distribution over the senses of the most likely K globals, use a self-attention score:
# query: the context representation: the average of the last C words, or the state of its own GRU)
# key vectors: the average context a sense appears in
# => we compute (q_c∙k_(s_i ))/√(d_k ) to get the self-attention logits, and the softmax to have a probability distribution over the senses of the selected K globals.
# We assign it, while keeping the other senses’ softmax at 10-8=~0, as usual
class SelfAtt(torch.nn.Module):

    def __init__(self, StandardLM, graph_dataobj, grapharea_size, grapharea_matrix,
                 vocabulary_df, batch_size, n_layers, n_hid_units, K, context_method, C, inputdata_folder, dim_qkv):

        super(SelfAtt, self).__init__()

        self.StandardLM = StandardLM
        init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                                     batch_size, n_layers, n_hid_units)

        # -------------------- Senses' architecture --------------------

        self.K = K
        self.grapharea_matrix_neighbours_section = self.grapharea_matrix[:, 0:self.grapharea_size]
        self.C = C

        precomputed_SC_filepath = os.path.join(inputdata_folder, str(C) + F.MATRIX_SENSE_CONTEXTS_FILEEND)
        senses_average_context_SC = np.load(precomputed_SC_filepath)
        self.SC = Parameter(torch.tensor(senses_average_context_SC, dtype=torch.float32), requires_grad=False)
        self.prev_word_embeddings = Parameter(torch.zeros((200, batch_size, self.SC.shape[1])), requires_grad=False)
        # prev_word_embeddings will be reshaped when we know the seq_len and self.batch_size

        init_context_handling(model=self, context_method=context_method)
        self.SelfAttLogits = ComputeLogits(dim_input_context=self.SC.shape[1], dim_input_elems=self.SC.shape[1],
                                           dim_qkv=dim_qkv, k=self.K, grapharea_size=self.grapharea_size)

    # ---------------------------------------- Forward call ----------------------------------------

    def forward(self, batchinput_tensor, batch_labels):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        # -------------------- Init --------------------
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn_context.detach_()
        self.location_context.detach_()
        if batchinput_tensor.shape[1] > 1:
            time_instants = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[1], dim=1)
        else:
            time_instants = [batchinput_tensor]

        word_embeddings_ls = []
        currentglobal_nodestates_ls = []

        # -------------------- Compute and collect input signals; predict globals -------------------
        for batch_elements_at_t in time_instants:
            get_input_signals(self, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls)

        word_embeddings = torch.stack(word_embeddings_ls, dim=0)
        global_nodestates = torch.stack(currentglobal_nodestates_ls,
                                        dim=0) if self.StandardLM.include_globalnode_input > 0 else None
        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)
        predictions_globals, _ = self.StandardLM(batchinput_tensor, batch_labels)

        # ------------------- Senses -------------------
        seq_len = batch_input_signals.shape[0]
        if self.predict_senses:

            # --------- Select the senses of the k most likely globals ---------
            k_globals_indices = predictions_globals.sort(descending=True).indices[:, 0:self.K]
            sample_k_indices_in_vocab_lls = k_globals_indices.tolist()

            all_sense_neighbours_ls = []
            sense_neighbours_len_ls = []
            for s in range(self.batch_size * seq_len):
                k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
                k_globals_lemmatized = [self.vocabulary_lemmatizedList[idx] for idx in k_globals_vocab_indices]

                sample_k_indices_lemmatized_ls = [
                    Utils.word_to_vocab_index(lemmatized_word, self.vocabulary_wordList) for
                    lemmatized_word in k_globals_lemmatized]
                sample_k_indices_lemmatized = torch.tensor(sample_k_indices_lemmatized_ls).to(CURRENT_DEVICE).squeeze(
                    dim=0) + self.last_idx_senses

                sense_neighbours = torch.tensor(get_senseneighbours_of_k_globals(self, sample_k_indices_lemmatized)) \
                    .squeeze(dim=0).to(torch.int64).to(CURRENT_DEVICE)
                sense_neighbours_len_ls.append(sense_neighbours.shape[0])

                random_sense_idx = np.random.randint(low=0, high=self.SC.shape[0])  # a random sense as pad/filler.
                all_sense_neighbours_ls.append(torch.nn.functional.pad(input=sense_neighbours, pad=[0,
                       self.K * self.grapharea_size - sense_neighbours.shape[0]], value=random_sense_idx))

            # -------- Senses: compute self-attention scores from context rep. and senses' average context --------
            # ----- Base for the artificial softmax -----
            senses_softmax = torch.ones((seq_len, self.batch_size, self.last_idx_senses)).to(CURRENT_DEVICE)
            epsilon = 10 ** (-8)
            senses_softmax = epsilon * senses_softmax  # base probability value for non-selected senses
            senses_mask = torch.zeros(size=(seq_len, self.batch_size, self.last_idx_senses)).to(torch.bool).to(
                CURRENT_DEVICE)

            # ----- update the location context with the latest word embeddings -----
            if self.context_method == ContextMethod.AVERAGE:
                update_context_average(self.location_context, word_embeddings, self.prev_word_embeddings, self.C,
                                       CURRENT_DEVICE)
                self.prev_word_embeddings.data = word_embeddings.data
            elif self.context_method == ContextMethod.GRU:
                context_out = rnn_loop(batch_input_signals, model=self, rnn_ls=self.context_rnn_ls,
                                       memory=self.memory_hn_context)
                self.location_context.data = context_out.clone()

            # ----- the context of the selected senses and the cosine similarity: -----
            senses_context = torch.zeros((self.location_context.shape[0], self.location_context.shape[1],
                                          self.grapharea_size * self.K, self.location_context.shape[2]))

            all_sense_neighbours = torch.stack(all_sense_neighbours_ls, dim=0).reshape(
                (seq_len, self.batch_size, self.grapharea_size * self.K))
            senses_context.data = self.SC[all_sense_neighbours, :].data

            # Insertion point
            sense_neighbours_logits_dupl = self.SelfAttLogits(input_q=self.location_context[0:seq_len], input_kv=senses_context)
            # writing in the artificial softmax
            for t in (range(seq_len)):
                for b in range(self.batch_size):
                    logits_senseneighbour_ls = list(set(zip(sense_neighbours_logits_dupl[t,b,:].tolist(), all_sense_neighbours[t,b,:].tolist())))
                    logits = sense_neighbours_logits_dupl[t,b,0:len(logits_senseneighbour_ls)]
                    senseneighbours = torch.tensor(all_sense_neighbours[t, b, :].tolist()[0:len(logits_senseneighbour_ls)]).to(CURRENT_DEVICE)
                    p_scores = torch.softmax(logits, dim=0)
                    senses_mask[t, b, senseneighbours] = True
                    senses_softmax[t,b].masked_scatter_(mask=senses_mask[t,b].data.clone(), source=p_scores)

            predictions_senses = torch.log(senses_softmax).reshape(seq_len * self.batch_size,
                                                                   senses_softmax.shape[2])

        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
