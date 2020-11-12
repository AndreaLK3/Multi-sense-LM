import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
from NN.Models.Common import init_model_parameters, init_common_architecture, predict_globals_withGRU, get_input_signals
from NN.Models.RNNSteps import rnn_loop
from torch.nn.parameter import Parameter
import Utils
from NN.Models.RNNSteps import reshape_memories


class RNN(torch.nn.Module):

    def __init__(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 include_globalnode_input, batch_size, n_layers, n_hid_units):

        # -------------------- Initialization in common: parameters & globals --------------------
        super(RNN, self).__init__()

        init_model_parameters(self, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, include_globalnode_input,
                              batch_size, n_layers, n_hid_units)
        init_common_architecture(self, embeddings_matrix, graph_dataobj)

        # -------------------- Senses' architecture --------------------
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))),
                                          requires_grad=False)
        self.senses_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=n_hid_units // 2 if i == n_layers - 1 else n_hid_units, num_layers=1)  # 512
             for i in range(n_layers)])

        self.linear2senses = torch.nn.Linear(in_features=n_hid_units // 2,  # 512
                                                 out_features=self.last_idx_senses, bias=True)

    # ---------------------------------------- Forward call ----------------------------------------

    def forward(self, batchinput_tensor):  # given the batches, the current node is at index 0
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        # -------------------- Init --------------------
        distributed_batch_size = batchinput_tensor.shape[0]
        if not (distributed_batch_size == self.batch_size) and not self.hidden_state_bsize_adjusted:
            reshape_memories(distributed_batch_size, self)

        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()
        self.memory_hn_senses.detach_()

        if batchinput_tensor.shape[1] > 1:
            time_instants = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[1], dim=1)
        else:
            time_instants = [batchinput_tensor]

        word_embeddings_ls = []
        currentglobal_nodestates_ls = []
        # -------------------- Compute input signals --------------------
        for batch_elements_at_t in time_instants:
            get_input_signals(self, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls)

        # -------------------- Collect input signals --------------------
        word_embeddings = torch.stack(word_embeddings_ls, dim=0) if self.include_globalnode_input<2 else None
        global_nodestates = torch.stack(currentglobal_nodestates_ls, dim=0) if self.include_globalnode_input>0 else None
        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # ------------------- Globals ------------------
        seq_len = batch_input_signals.shape[0]
        predictions_globals, _logits_globals = predict_globals_withGRU(self, batch_input_signals, seq_len, distributed_batch_size)

        # ------------------- Senses -------------------
        if self.predict_senses:
            task_2_out = rnn_loop(batch_input_signals, model=self, rnn_ls=self.senses_rnn_ls, memory=self.memory_hn_senses)
            task2_out = task_2_out.reshape(distributed_batch_size * seq_len, task_2_out.shape[2])

            logits_sense = self.linear2senses(task2_out)
            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
