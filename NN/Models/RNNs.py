import torch
from torch_geometric.nn import GATConv
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
import torch.nn.functional as tfunc
from NN.Models.Common import unpack_input_tensor, init_model_parameters, lemmatize_node
from NN.Models.Steps_RNN import rnn_loop
from torch.nn.parameter import Parameter
import Utils
from PrepareKBInput.LemmatizeNyms import lemmatize_term
from NN.Models.Steps_RNN import reshape_memories, select_layer_memory, update_layer_memory

def run_graphnet(t_input_lts, batch_elems_at_t,t_globals_indices_ls, CURRENT_DEVICE, model):
    graph_batch_ls = []
    current_location_in_batchX_ls = []
    rows_to_skip = 0
    if model.include_globalnode_input:
        t_edgeindex_g_ls = [t_input_lts[b][0][1] for b in range(len(t_input_lts))]
        t_edgetype_g_ls = [t_input_lts[b][0][2] for b in range(len(t_input_lts))]

        for i_sample in range(batch_elems_at_t.shape[0]):
            sample_edge_index = t_edgeindex_g_ls[i_sample]
            sample_edge_type = t_edgetype_g_ls[i_sample]
            x_indices, edge_index, edge_type = lemmatize_node(t_globals_indices_ls[i_sample], sample_edge_index, sample_edge_type, model=model)
            sample_x = model.X.index_select(dim=0, index=x_indices.squeeze())

            currentword_location_in_batchX = rows_to_skip + current_location_in_batchX_ls[-1] \
                if len(current_location_in_batchX_ls) > 0 else 0
            rows_to_skip = sample_x.shape[0]
            current_location_in_batchX_ls.append(currentword_location_in_batchX)

            sample_graph = Data(x=sample_x, edge_index=sample_edge_index)
            graph_batch_ls.append(sample_graph)

        batch_graph = Batch.from_data_list(graph_batch_ls)
        x_attention_states = model.gat_globals(batch_graph.x, batch_graph.edge_index)
        t_currentglobal_node_states = x_attention_states.index_select(dim=0, index=torch.tensor(
            current_location_in_batchX_ls).to(torch.int64).to(CURRENT_DEVICE))
        return t_currentglobal_node_states


class RNN(torch.nn.Module):

    def __init__(self, model_type, data, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                 include_globalnode_input,
                 batch_size, n_layers, n_hid_units, dropout_p):

        # -------------------- Initialization and parameters --------------------
        super(RNN, self).__init__()
        self.model_type = model_type  # can be "LSTM" or "GRU"
        init_model_parameters(self, data, grapharea_size, grapharea_matrix, vocabulary_df, include_globalnode_input,
                                   batch_size, n_layers, n_hid_units, dropout_p)


        self.E = Parameter(embeddings_matrix.clone().detach(), requires_grad=True) # The matrix of embeddings
        self.dim_embs = self.E.shape[1]
        self.X = Parameter(data.x.clone().detach(), requires_grad=True)  # The matrix of global-nodestates

        # -------------------- Utilities --------------------
        # utility tensors, used in index_select etc.
        self.select_first_indices = Parameter(torch.tensor(list(range(n_hid_units))).to(torch.float32),requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        # Memories of the hidden states; overwritten at the 1st forward, when we know the distributed batch size
        self.memory_hn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_cn = Parameter(torch.zeros(size=(n_layers, batch_size, n_hid_units)), requires_grad=False)
        self.memory_hn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))), requires_grad=False)
        self.memory_cn_senses = Parameter(torch.zeros(size=(n_layers, batch_size, int(n_hid_units))), requires_grad=False)
        self.hidden_state_bsize_adjusted = False

        # -------------------- Input signals --------------------
        self.concatenated_input_dim = self.dim_embs + int(include_globalnode_input) * Utils.GRAPH_EMBEDDINGS_DIM
        # GAT for the node-states from the dictionary graph
        if self.include_globalnode_input:
            self.gat_globals = GATConv(in_channels=Utils.GRAPH_EMBEDDINGS_DIM, out_channels=int(Utils.GRAPH_EMBEDDINGS_DIM / 2),
                                       heads=2)  # , node_dim=1)
            # lemmatize_term('init', self.lemmatizer)# to establish LazyCorpusLoader and prevent a multi-thread crash

        # -------------------- The networks --------------------
        self.main_rnn_ls = torch.nn.ModuleList(
            [getattr(torch.nn, self.model_type)(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=450 if i == n_layers - 1 else n_hid_units, num_layers=1) for
             i in range(n_layers)])

        self.senses_rnn_ls = torch.nn.ModuleList(
            [getattr(torch.nn, self.model_type)(input_size=self.concatenated_input_dim if i == 0 else n_hid_units,
                                                hidden_size=450 if i == n_layers - 1 else n_hid_units, num_layers=1)
             for i in range(n_layers)])

        # 2nd part of the network: 2 linear layers to the logits
        self.linear2global = torch.nn.Linear(in_features=450,  #
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        self.linear2senses = torch.nn.Linear(in_features=450,  #
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
        self.memory_cn.detach_()
        self.memory_hn_senses.detach_()
        self.memory_cn_senses.detach_()

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
                t_g_nodestates = run_graphnet(t_input_lts, batch_elems_at_t,t_globals_indices_ls, CURRENT_DEVICE, self)
                currentglobal_nodestates_ls.append(t_g_nodestates)

        word_embeddings = torch.stack(word_embeddings_ls, dim=0)
        global_nodestates = torch.stack(currentglobal_nodestates_ls, dim=0) if self.include_globalnode_input else None

        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))  # , currentsense_node_state]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # ------------------- Globals -------------------
        # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
        task_1_out = rnn_loop(batch_input_signals, model=self)  # self.network_1_L1(input)
        task_1_out = task_1_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)

        seq_len = batch_input_signals.shape[0]
        task_1_out = task_1_out.reshape(distributed_batch_size * seq_len, task_1_out.shape[2])

        logits_global = self.linear2global(task_1_out)  # shape=torch.Size([5])
        predictions_globals = tfunc.log_softmax(logits_global, dim=1)

        # ------------------- Senses -------------------
        # line 1: GRU for senses + linear FF-NN tos logits.
        if self.predict_senses:
            task_2_out = rnn_loop(batch_input_signals, model=self)
            task2_out = task_2_out.reshape(distributed_batch_size * seq_len, task_2_out.shape[2])

            logits_sense = self.linear2senses(task2_out)
            predictions_senses = tfunc.log_softmax(logits_sense, dim=1)
        else:
            predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)

        return predictions_globals, predictions_senses

