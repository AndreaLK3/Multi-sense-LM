import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Models.Variants.Common as Common
from Models.Variants.RNNSteps import rnn_loop
from torch.nn.parameter import Parameter
import Utils
import logging
import Models.StandardLM.MiniTransformerXL as TXL

##### Everything related to the standard language model on words, identical across model variants, is collected here
##### This constitutes the class StandardLM: a component of the model variants (wrappers) that also address the senses

##### 1.1: Initializing parameters
def init_standardLM_model_parameters(model, graph_dataobj, grapharea_size, model_type, include_globalnode_input, batch_size):
    model.grapharea_size = grapharea_size
    model.include_globalnode_input = include_globalnode_input

    model.graph_dataobj = graph_dataobj
    model.last_idx_senses = graph_dataobj.node_types.tolist().index(1)
    model.last_idx_globals = graph_dataobj.node_types.tolist().index(2)

    model.batch_size = batch_size

    if model_type == 'gru':
        model.use_gold_lm = False
        model.use_transformer_lm = False
    elif model_type == 'transformer':
        model.use_gold_lm = False
        model.use_transformer_lm = True
    elif model_type == "gold_lm":
        model.use_gold_lm = True
        model.use_transformer_lm = False

##### 1.2: Initialization of: E, X, globals' (i.e. main) RNN / transformer
def init_common_architecture(model, embeddings_matrix, graph_dataobj):
    model.E = Parameter(embeddings_matrix.clone().detach(), requires_grad=True) # The matrix of embeddings
    model.dim_embs = model.E.shape[1]
    if model.include_globalnode_input > 0:
        model.X = Parameter(graph_dataobj.x.clone().detach(), requires_grad=True)  # The graph matrix

    # -------------------- Input signals --------------------
    model.concatenated_input_dim = model.dim_embs + Utils.GRAPH_EMBEDDINGS_DIM if model.include_globalnode_input==1 \
        else model.dim_embs # i.e. if 0 (no graph)

    # GAT for the node-states from the dictionary graph
    if model.include_globalnode_input ==1:
        model.gat_globals = GATConv(in_channels=Utils.GRAPH_EMBEDDINGS_DIM , out_channels=int(Utils.GRAPH_EMBEDDINGS_DIM / 2),
                                   heads=2)

    # -------------------- The networks --------------------
    if not model.use_gold_lm:
        if model.use_transformer_lm:
            model.standard_lm_transformer = TXL.get_mini_txl_modelobj() # pre-defined model parameters: 12 layers, etc.
        else:
            model.hidden_size = 1024
            model.n_layers = 3
            model.standard_rnn_ls = torch.nn.ModuleList(
                [torch.nn.GRU(input_size=model.concatenated_input_dim if i == 0 else model.hidden_size,
                                                    hidden_size=model.hidden_size // 2 if i == model.n_layers - 1 else model.hidden_size, num_layers=1)  # 512
                 for i in range(model.n_layers)])
            model.linear2global = torch.nn.Linear(in_features=model.hidden_size // 2,  # 512
                                         out_features=model.last_idx_globals - model.last_idx_senses, bias=True)

            model.memory_hn = Parameter(torch.zeros(size=(model.n_layers, model.batch_size, model.hidden_size)),
                                    requires_grad=False)
            model.select_first_indices = Parameter(torch.tensor(list(range(2 * model.hidden_size))).to(torch.float32),
                                                   requires_grad=False)


##### 1.4: get_input_signals (found in Models.Variants.Common)

##### 1.5: standard LM prediction with globals
# --- 1.5a: standard LM with a GRU
def predict_globals_withGRU(model, batch_input_signals, seq_len, batch_size):

    # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
    task_1_out = rnn_loop(batch_input_signals, model, model.standard_rnn_ls, model.memory_hn)  # self.network_1_L1(input)
    task_1_out = task_1_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)

    task_1_out = task_1_out.reshape(batch_size * seq_len, task_1_out.shape[2])

    logits_global = model.linear2global(task_1_out)  # shape=torch.Size([5])
    predictions_globals = tfunc.log_softmax(logits_global, dim=1)

    return predictions_globals, logits_global


# --- 1.5b: alternative: standardLM with a Transformer-XL architecture
def predict_globals_withTXL(model, batch_input_signals, seq_len, batch_size):
    return model.standard_lm_transformer(batch_input_signals)


#####
class StandardLM(torch.nn.Module):

    def __init__(self, graph_dataobj, grapharea_size, embeddings_matrix, model_type, include_graph_input, batch_size):

        # -------------------- Initialization in common: parameters & globals --------------------
        super(StandardLM, self).__init__()

        init_standardLM_model_parameters(self, graph_dataobj, grapharea_size, model_type, include_graph_input, batch_size)
        init_common_architecture(self, embeddings_matrix, graph_dataobj)


    def forward(self, batch_input_tensor, batch_labels):
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()

        # ---------- Organizing input, computing input signals ----------
        input_batch_size = batch_input_tensor.shape[0]
        if input_batch_size != self.batch_size: # in case of batch size mismatch between pre-training and training
            self.batch_size = input_batch_size
            self.memory_hn = Parameter(torch.rand(size=(self.n_layers, self.batch_size, self.hidden_size)).to(CURRENT_DEVICE),
                                        requires_grad=False)

        seq_len = batch_input_tensor.shape[1]
        if batch_input_tensor.shape[1] > 1:
            time_instants = torch.chunk(batch_input_tensor, chunks=batch_input_tensor.shape[1], dim=1)
        else:
            time_instants = [batch_input_tensor]
        word_embeddings_ls = []
        currentglobal_nodestates_ls = []

        for batch_elements_at_t in time_instants:
            Common.get_input_signals(self, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls)

        # -------------------- Collecting input signals --------------------
        word_embeddings = torch.stack(word_embeddings_ls, dim=0) if self.include_globalnode_input<2 else None
        global_nodestates = torch.stack(currentglobal_nodestates_ls, dim=0) if self.include_globalnode_input>0 else None
        batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                             [word_embeddings, global_nodestates]))
        batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)

        # ---------- Predicting the next word ----------
        if self.use_gold_lm:
            predictions_globals = Common.assign_one(batch_labels[:, 0], seq_len, self.batch_size,
                                                    self.last_idx_globals - self.last_idx_senses, CURRENT_DEVICE)
            return predictions_globals

        if self.use_transformer_lm:
             predictions_globals = predict_globals_withTXL(self, batch_input_signals, seq_len, self.batch_size)
        else:
            predictions_globals, _logits_globals = predict_globals_withGRU(self, batch_input_signals, seq_len,
                                                                               self.batch_size)
        # placeholder for the senses, to be able to use the same training facilities as the model variants
        predictions_senses = torch.tensor([0] * self.batch_size * seq_len).to(CURRENT_DEVICE)
        return predictions_globals, predictions_senses




