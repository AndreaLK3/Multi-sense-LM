import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
from NN.Models.Common import unpack_input_tensor, init_model_parameters, run_graphnet
from NN.Models.Steps_RNN import rnn_loop
from torch.nn.parameter import Parameter
import Utils
from NN.Models.Steps_RNN import reshape_memories


class RNN(torch.nn.Module):

    def __init__(self, data, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 include_globalnode_input, batch_size, n_layers, n_hid_units):

        # -------------------- Initialization and parameters --------------------
        super(RNN, self).__init__()
        init_model_parameters(self, data, grapharea_size, grapharea_matrix, vocabulary_df, include_globalnode_input,
                                   batch_size, n_layers, n_hid_units)

        self.E = Parameter(embeddings_matrix.clone().detach(), requires_grad=True) # The matrix of embeddings
        self.dim_embs = self.E.shape[1]
        if include_globalnode_input:
            self.X = Parameter(data.x.clone().detach(), requires_grad=True)  # The graph matrix

        # -------------------- Utilities --------------------
        # utility tensors, used in index_select etc.
        self.select_first_indices = Parameter(torch.tensor(list(range(n_hid_units))).to(torch.float32),requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        # Memories of the hidden states; overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))), requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # -------------------- Input signals --------------------
        self.concatenated_input_dim = self.dim_embs + int(include_globalnode_input) * Utils.GRAPH_EMBEDDINGS_DIM
        # GAT for the node-states from the dictionary graph
        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=Utils.GRAPH_EMBEDDINGS_DIM, out_channels=int(Utils.GRAPH_EMBEDDINGS_DIM / 2),
                                       heads=2)  # , node_dim=1)

        # -------------------- The networks --------------------
        self.main_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=n_hid_units // 2 if i == n_layers - 1 else n_hid_units, num_layers=1)  # 512
             for i in range(n_layers)])

        self.senses_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=n_hid_units // 2 if i == n_layers - 1 else n_hid_units, num_layers=1)  # 512
             for i in range(n_layers)])

        # 2nd part of the network: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=n_hid_units // 2,  # 512
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
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

        for batch_elements_at_t in time_instants:
            batch_elems_at_t = batch_elements_at_t.squeeze(dim=1)
            elems_at_t_ls = batch_elements_at_t.chunk(chunks=batch_elems_at_t.shape[0], dim=0)

            t_input_lts = [unpack_input_tensor(sample_tensor, self.grapharea_size) for sample_tensor in elems_at_t_ls]
            t_globals_indices_ls = [t_input_lts[b][0][0] for b in range(len(t_input_lts))]

            # -------------------- Input --------------------
            # Input signal n.1: the embedding of the current (global) word
            t_current_globals_indices_ls = [x_indices[0]-self.last_idx_senses for x_indices in t_globals_indices_ls]
            t_current_globals_indices = torch.stack(t_current_globals_indices_ls, dim=0)
            t_word_embeddings = self.E.index_select(dim=0, index=t_current_globals_indices)
            word_embeddings_ls.append(t_word_embeddings)
            # Input signal n.2: the node-state of the current global word - now with graph batching
            if self.include_globalnode_input:
                t_g_nodestates = run_graphnet(t_input_lts, batch_elems_at_t, t_globals_indices_ls, CURRENT_DEVICE, self)
                currentglobal_nodestates_ls.append(t_g_nodestates)

        word_embeddings = torch.stack(word_embeddings_ls, dim=0)
        global_nodestates = torch.stack(currentglobal_nodestates_ls, dim=0) if self.include_globalnode_input else None

        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))  # , currentsense_node_state]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # ------------------- Globals -------------------
        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        task_1_out = rnn_loop(batch_input_signals, model=self, globals_or_senses_rnn=True)  # self.network_1_L1(input)
        task_1_out = task_1_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)

        seq_len = batch_input_signals.shape[0]
        task_1_out = task_1_out.reshape(distributed_batch_size * seq_len, task_1_out.shape[2])

        logits_global = self.linear2global(task_1_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)

        # ------------------- Senses -------------------
        # line 1: GRU for senses + linear FF-NN to logits.
        if self.predict_senses:
            task_2_out = rnn_loop(batch_input_signals, model=self, globals_or_senses_rnn=False)
            task2_out = task_2_out.reshape(distributed_batch_size * seq_len, task_2_out.shape[2])

            logits_sense = self.linear2senses(task2_out)
            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses




class RNN_on_GAT(torch.nn.Module):

    def __init__(self, data, grapharea_size, grapharea_matrix, vocabulary_df, batch_size, n_layers, n_hid_units,
                 concat_input_dim, num_heads, dropout_p):

        # -------------------- Initialization and parameters --------------------
        super(RNN_on_GAT, self).__init__()
        init_model_parameters(self, data, grapharea_size, grapharea_matrix, vocabulary_df, include_globalnode_input=True,
                                   batch_size=batch_size, n_layers=n_layers, n_hid_units=n_hid_units, dropout_p=dropout_p)
        self.num_heads = num_heads
        self.concatenated_input_dim = concat_input_dim

        # self.E = Parameter(embeddings_matrix.clone().detach(), requires_grad=True) # The matrix of embeddings
        # FastText embeddings are not used in this version, where the only input signal is the GAT nodestate

        # if include_globalnode_input:
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)  # The matrix of global-nodestates

        # -------------------- Utilities --------------------
        # utility tensors, used in index_select etc.
        self.select_first_indices = Parameter(torch.tensor(list(range(n_hid_units))).to(torch.float32),requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.concatenated_input_dim)), requires_grad=False)

        # Memories of the hidden states; overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))), requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # -------------------- Input signals --------------------
        # GAT for the node-states from the dictionary graph
        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=Utils.GRAPH_EMBEDDINGS_DIM, out_channels=self.concatenated_input_dim,
                                       heads=self.num_heads)  # , node_dim=1)

        # -------------------- The networks --------------------
        self.main_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=n_hid_units // 2 if i == n_layers - 1 else n_hid_units, num_layers=1)  # 512
             for i in range(n_layers)])

        self.senses_rnn_ls = torch.nn.ModuleList(
            [torch.nn.GRU(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=n_hid_units // 2 if i == n_layers - 1 else n_hid_units, num_layers=1)  # 512
             for i in range(n_layers)])

        # 2nd part of the network: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=n_hid_units // 2,  # 512
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
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

        for batch_elements_at_t in time_instants:
            batch_elems_at_t = batch_elements_at_t.squeeze(dim=1)
            elems_at_t_ls = batch_elements_at_t.chunk(chunks=batch_elems_at_t.shape[0], dim=0)

            t_input_lts = [unpack_input_tensor(sample_tensor, self.grapharea_size) for sample_tensor in elems_at_t_ls]
            t_globals_indices_ls = [t_input_lts[b][0][0] for b in range(len(t_input_lts))]

            # -------------------- Input --------------------
            # Input signal n.1: the embedding of the current (global) word
            t_current_globals_indices_ls = [x_indices[0]-self.last_idx_senses for x_indices in t_globals_indices_ls]
            t_current_globals_indices = torch.stack(t_current_globals_indices_ls, dim=0)
            t_word_embeddings = self.E.index_select(dim=0, index=t_current_globals_indices)
            word_embeddings_ls.append(t_word_embeddings)
            # Input signal n.2: the node-state of the current global word - now with graph batching
            if self.include_globalnode_input:
                t_g_nodestates = run_graphnet(t_input_lts, batch_elems_at_t, t_globals_indices_ls, CURRENT_DEVICE, self)
                currentglobal_nodestates_ls.append(t_g_nodestates)

        word_embeddings = torch.stack(word_embeddings_ls, dim=0)
        global_nodestates = torch.stack(currentglobal_nodestates_ls, dim=0) if self.include_globalnode_input else None

        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))  # , currentsense_node_state]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # ------------------- Globals -------------------
        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        task_1_out = rnn_loop(batch_input_signals, model=self, globals_or_senses_rnn=True)  # self.network_1_L1(input)
        task_1_out = task_1_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)

        seq_len = batch_input_signals.shape[0]
        task_1_out = task_1_out.reshape(distributed_batch_size * seq_len, task_1_out.shape[2])

        logits_global = self.linear2global(task_1_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)

        # ------------------- Senses -------------------
        # line 1: GRU for senses + linear FF-NN to logits.
        if self.predict_senses:
            task_2_out = rnn_loop(batch_input_signals, model=self, globals_or_senses_rnn=False)
            task2_out = task_2_out.reshape(distributed_batch_size * seq_len, task_2_out.shape[2])

            logits_sense = self.linear2senses(task2_out)
            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses
