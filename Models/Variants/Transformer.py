import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc

import Models.Variants.Common
import Models.Variants.Common as Common
import Models.Variants.InputSignals as InputSignals
from Models.Variants.RNNSteps import rnn_loop
import Utils
from torch.nn.parameter import Parameter
import Models.StandardLM.MiniTransformerXL as TXL


class Transformer(torch.nn.Module):

    def __init__(self, StandardLM, graph_dataobj, grapharea_size, grapharea_matrix,
                 vocabulary_df, batch_size, n_layers, n_hid_units):

        super(Transformer, self).__init__()

        self.StandardLM = StandardLM
        Common.init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                                     batch_size, n_layers, n_hid_units)

        # -------------------- Senses' architecture --------------------
        self.TransformerForSenses = TXL.get_mini_txl_modelobj(self.last_idx_senses)
        self.memsForSenses = None

    # ---------------------------------------- Forward call ----------------------------------------

    def forward(self, batchinput_tensor, batch_labels):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        # -------------------- Init --------------------
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        # self.memory_hn_senses.detach_()
        if batchinput_tensor.shape[1] > 1:
            time_instants = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[1], dim=1)
        else:
            time_instants = [batchinput_tensor]

        word_embeddings_ls = []
        currentglobal_nodestates_ls = []
        globals_input_ids_ls = []  # for the transformer

        # -------------------- Compute and collect input signals; predict globals -------------------
        for batch_elements_at_t in time_instants:
            InputSignals.get_input_signals(self, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls,
                      globals_input_ids_ls)

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
            input_indices = torch.cat(globals_input_ids_ls, dim=0).reshape((seq_len, self.batch_size)).permute(1, 0)
            predictions_senses, mems = Common.predict_withTXL(self.TransformerForSenses, self.memsForSenses, input_indices)
            self.memsForSenses = mems
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses