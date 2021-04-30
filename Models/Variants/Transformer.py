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

        batch_input_signals, globals_input_ids_ls, word_embeddings, predictions_globals = \
            Common.get_input_and_predict_globals(self, batchinput_tensor, batch_labels)

        # ------------------- Senses -------------------
        seq_len = batch_input_signals.shape[0]
        if self.predict_senses:
            input_indices = torch.cat(globals_input_ids_ls, dim=0).reshape((seq_len, self.batch_size)).permute(1, 0)
            predictions_senses, mems = Common.predict_withTXL(self.TransformerForSenses, self.memsForSenses, input_indices)
            self.memsForSenses = mems
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses