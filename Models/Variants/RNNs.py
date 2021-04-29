import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Models.Variants.Common as Common
import Models.Variants.InputSignals
from Models.Variants.RNNSteps import rnn_loop
import Utils
from torch.nn.parameter import Parameter


class RNN(torch.nn.Module):

    def __init__(self, StandardLM, graph_dataobj, grapharea_size, grapharea_matrix,
                 vocabulary_df, batch_size, n_layers, n_hid_units):

        super(RNN, self).__init__()

        self.StandardLM = StandardLM
        Common.init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                                     batch_size, n_layers, n_hid_units)

        # -------------------- Senses' architecture --------------------
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))),
                                          requires_grad=False)
        self.senses_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.StandardLM.concatenated_input_dim if i == 0 else n_hid_units,
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

        # -------------------- Compute and collect input signals; predict globals -------------------
        for batch_elements_at_t in time_instants:
            Models.Variants.InputSignals.get_input_signals(self, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls)

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
            task_2_out = rnn_loop(batch_input_signals, model=self, rnn_ls=self.senses_rnn_ls, memory=self.memory_hn_senses)
            task2_out = task_2_out.reshape(self.batch_size * seq_len, task_2_out.shape[2])

            logits_sense = self.linear2senses(task2_out)
            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
