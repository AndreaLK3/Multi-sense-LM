import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as tfunc
import Models.Variants.Common as Common
from Models.Variants.RNNSteps import rnn_loop
from torch.nn.parameter import Parameter
import Utils
import Models.StandardLM.MiniTransformerXL as TXL

##### Everything related to the standard language model on words, identical across model variants, is collected here
##### This constitutes the class StandardLM: a component of the model variants (wrappers) that also address the senses

##### 1.1: Initializing parameters
def init_standardLM_model_parameters(model, grapharea_size,
                                     include_globalnode_input, use_gold_lm, use_transformer_lm):
    model.grapharea_size = grapharea_size
    model.include_globalnode_input = include_globalnode_input
    model.use_gold_lm = use_gold_lm
    model.use_transformer_lm = use_transformer_lm


##### 1.2: Initialization of: E, X, globals' (i.e. main) RNN / transformer
def init_common_architecture(model, embeddings_matrix, graph_dataobj):
    model.E = Parameter(embeddings_matrix.clone().detach(), requires_grad=True) # The matrix of embeddings
    model.dim_embs = model.E.shape[1]
    if model.include_globalnode_input > 0:
        model.X = Parameter(graph_dataobj.x.clone().detach(), requires_grad=True)  # The graph matrix

    model.memory_hn = Parameter(torch.zeros(size=(model.n_layers, model.batch_size, model.hidden_size)), requires_grad=False)

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

    def __init__(self, graph_dataobj, grapharea_size, embeddings_matrix,
                 include_globalnode_input, use_gold_lm, use_transformer_lm):

        # -------------------- Initialization in common: parameters & globals --------------------
        super(StandardLM, self).__init__()

        init_standardLM_model_parameters(self, grapharea_size, include_globalnode_input, use_gold_lm, use_transformer_lm)
        init_common_architecture(self, embeddings_matrix, graph_dataobj)


    def forward(self, batch_input_signals, batch_labels):
        CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())

        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_hn.detach_()

        # ------------------- Globals ------------------
        seq_len = batch_input_signals.shape[0]
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




