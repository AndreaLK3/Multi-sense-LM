import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Graph.Adjacencies as AD
import Models.Variants.Common as Common
from Models.Variants.RNNSteps import rnn_loop, reshape_memories
from Utils import DEVICE
from torch.nn.parameter import Parameter
import logging

# ****** Auxiliary functions *******
def get_senseneighbours_of_k_globals(model, sample_k_indices):

    sample_neighbours_section = (model.grapharea_matrix[sample_k_indices.cpu(), 0:32].todense()) -1
    sample_neighbours = sample_neighbours_section[sample_neighbours_section >= 0]
    sense_neighbours = sample_neighbours[sample_neighbours < model.last_idx_senses]

    # fix for rare error (1 every several thousands of tokens), where we find no senses:
    if sense_neighbours.size == 0:
        sense_neighbours = (torch.rand((1)) * model.last_idx_senses).unsqueeze(0)
        logging.debug("Found 0 senses for word_idx=" + str(sample_k_indices) + ", using random")

    return sense_neighbours


def subtract_probability_mass_from_selected(softmax_selected_senses, delta_to_subtract):
    try:
        max_index_t = torch.argmax(softmax_selected_senses)
        prev_max_value = softmax_selected_senses[max_index_t]
        softmax_selected_senses[max_index_t].data = prev_max_value - delta_to_subtract
    except RuntimeError as e:
        logging.info("softmax_selected_senses.shape=" + str(softmax_selected_senses.shape))
        logging.info("delta_to_subtract=" + str(delta_to_subtract))
        raise e

    return softmax_selected_senses

# *****************************

# Choose among the senses of the most likely k=,1,5,10... globals.
# Add the [the probability distribution over those] to [the 0 distribution over the whole senses' vocabulary]...

class SelectK(torch.nn.Module):

    def __init__(self, StandardLM, graph_dataobj, grapharea_size, grapharea_matrix,
                 vocabulary_df, batch_size, n_layers, n_hid_units, K):

        super(SelectK, self).__init__()

        self.StandardLM = StandardLM
        Common.init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                                     batch_size, n_layers, n_hid_units)

        # -------------------- Senses' architecture --------------------
        self.K = K
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

    def forward(self, batchinput_tensor, batch_labels):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        # -------------------- Init --------------------
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn_senses.detach_()

        if batchinput_tensor.shape[1] > 1:
            time_instants = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[1], dim=1)
        else:
            time_instants = [batchinput_tensor]

        word_embeddings_ls = []
        currentglobal_nodestates_ls = []

        # -------------------- Compute and collect input signals -------------------
        for batch_elements_at_t in time_instants:
            Common.get_input_signals(self, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls)

        word_embeddings = torch.stack(word_embeddings_ls, dim=0) if self.include_globalnode_input < 2 else None
        global_nodestates = torch.stack(currentglobal_nodestates_ls,
                                        dim=0) if self.include_globalnode_input > 0 else None
        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        predictions_globals = self.StandardLM(batch_input_signals, batch_labels)

        # ------------------- Senses -------------------
        # line 1: GRU for senses + linear FF-Models to logits.
        seq_len = batch_input_signals.shape[0]
        if self.predict_senses:
            task_2_out = rnn_loop(batch_input_signals, model=self, rnn_ls=self.senses_rnn_ls, memory=self.memory_hn_senses)
            task2_out = task_2_out.reshape(self.batch_size * seq_len, task_2_out.shape[2])
            logits_senses = self.linear2senses(task2_out)

            # line 2: select senses of the k most likely globals
            k_globals_indices = predictions_globals.sort(descending=True).indices[:, 0:self.K]

            senses_softmax = torch.ones((self.batch_size * seq_len, self.last_idx_senses)).to(CURRENT_DEVICE)
            epsilon = 10 ** (-8)
            senses_softmax = epsilon * senses_softmax  # base probability value for non-selected senses
            i_senseneighbours_mask = torch.zeros(size=(self.batch_size * seq_len, self.last_idx_senses)).to(torch.bool).to(CURRENT_DEVICE)

            sample_k_indices_in_vocab_lls = k_globals_indices.tolist()

            for s in range(self.batch_size * seq_len):

                k_globals_vocab_indices = sample_k_indices_in_vocab_lls[s]
                k_globals_lemmatized = [self.vocabulary_lemmatizedList[idx] for idx in k_globals_vocab_indices]

                sample_k_indices_lemmatized_ls = self.vocabulary_df[self.vocabulary_df['word'].isin(k_globals_lemmatized)].index.to_list()

                sample_k_indices_lemmatized = torch.tensor(sample_k_indices_lemmatized_ls).to(CURRENT_DEVICE).squeeze(dim=0) + self.last_idx_senses

                sense_neighbours_t = torch.tensor(get_senseneighbours_of_k_globals(self, sample_k_indices_lemmatized))\
                    .squeeze(dim=0).to(torch.int64).to(CURRENT_DEVICE)

                # standard procedure: get the logits of the senses of the most likely globals,
                # apply a softmax only over them, and then assign an epsilon probability to the other senses
                sample_logits_senses = logits_senses.index_select(dim=0, index=self.select_first_indices[s].to(torch.int64)).squeeze()
                logits_selected_senses = sample_logits_senses.index_select(dim=0, index=sense_neighbours_t)
                softmax_selected_senses = tfunc.softmax(input=logits_selected_senses, dim=0)

                # for i in range(len(sense_neighbours_t)):
                #     i_senseneighbours_mask[s,sense_neighbours_t[i]]=True
                i_senseneighbours_mask[s, sense_neighbours_t] = True

                quantity_to_subtract_from_selected = epsilon * (self.last_idx_senses - len(sense_neighbours_t))
                softmax_selected_senses = subtract_probability_mass_from_selected(softmax_selected_senses, quantity_to_subtract_from_selected)
                senses_softmax[s].masked_scatter_(mask=i_senseneighbours_mask[s].data.clone(), source=softmax_selected_senses)

                # Utils.log_chronometer([t0, t1, t2, t3, t4, t5, t6, t7])
            predictions_senses = torch.log(senses_softmax)

        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses