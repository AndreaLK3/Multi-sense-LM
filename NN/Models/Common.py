import logging
from enum import Enum

import torch
from torch.nn import functional as tfunc
from torch.nn.parameter import Parameter

from Graph import Adjacencies as AD
from Utils import DEVICE
import Utils
from torch_geometric.nn import GATConv
from NN.Models.RNNSteps import rnn_loop


#########################
##### 1: Model steps ####C
#########################

##### 1.1: Initialization of: graph_dataobj, grapharea_matrix, vocabulary_lists & more
def init_model_parameters(model, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                          include_globalnode_input,
                          batch_size, n_layers, n_hid_units):
    model.grapharea_matrix = grapharea_matrix

    model.vocabulary_df = vocabulary_df
    model.vocabulary_wordList = vocabulary_df['word'].to_list().copy()
    model.vocabulary_lemmatizedList = vocabulary_df['lemmatized_form'].to_list().copy()

    model.include_globalnode_input = include_globalnode_input
    model.predict_senses = False # it can be set to True when starting a training loop

    model.last_idx_senses = graph_dataobj.node_types.tolist().index(1)
    model.last_idx_globals = graph_dataobj.node_types.tolist().index(2)

    model.grapharea_size = grapharea_size

    model.dim_embs = graph_dataobj.x.shape[1]
    model.batch_size = batch_size
    model.n_layers = n_layers
    model.hidden_size = n_hid_units

    return

##### 1.2: Initialization of: E, X, globals' (i.e. main) RNN
def init_common_architecture(model, embeddings_matrix, graph_dataobj):
    model.E = Parameter(embeddings_matrix.clone().detach(), requires_grad=True) # The matrix of embeddings
    model.dim_embs = model.E.shape[1]
    if model.include_globalnode_input:
        model.X = Parameter(graph_dataobj.x.clone().detach(), requires_grad=True)  # The graph matrix

    # -------------------- Utilities --------------------
    # utility tensors, used in index_select etc.
    model.select_first_indices = Parameter(torch.tensor(list(range(2*model.hidden_size))).to(torch.float32),requires_grad=False)
    model.embedding_zeros = Parameter(torch.zeros(size=(1, model.dim_embs)), requires_grad=False)

    # Memories of the hidden states; overwritten at the 1st forward, when we know the distributed batch size
    model.memory_hn = Parameter(torch.zeros(size=(model.n_layers, model.batch_size, model.hidden_size)), requires_grad=False)
    model.hidden_state_bsize_adjusted = False

    # -------------------- Input signals --------------------
    model.concatenated_input_dim = model.dim_embs + int(model.include_globalnode_input) * Utils.GRAPH_EMBEDDINGS_DIM
    # GAT for the node-states from the dictionary graph
    if model.include_globalnode_input:
        model.gat_globals = GATConv(in_channels=Utils.GRAPH_EMBEDDINGS_DIM, out_channels=int(Utils.GRAPH_EMBEDDINGS_DIM / 2),
                                   heads=2)  # , node_dim=1)

    # -------------------- The networks --------------------½
    model.main_rnn_ls = torch.nn.ModuleList(
        [torch.nn.GRU(input_size=model.concatenated_input_dim if i == 0 else model.hidden_size,
                                            hidden_size=model.hidden_size // 2 if i == model.n_layers - 1 else model.hidden_size, num_layers=1)  # 512
         for i in range(model.n_layers)])
    model.linear2global = torch.nn.Linear(in_features=model.hidden_size // 2,  # 512
                                         out_features=model.last_idx_globals - model.last_idx_senses, bias=True)

##### 1.3: Extracting the input elements (x_indices, edge_index, edge_type) from the padded tensor in the batch
def unpack_to_input_tpl(in_tensor, grapharea_size, max_edges):
    x_indices = in_tensor[(in_tensor[0:grapharea_size] != -1).nonzero().flatten()]
        # shortcut for the case when there is no sense
    if x_indices.nonzero().shape[0] == 0:
        edge_index = torch.zeros(size=(2,max_edges)).to(DEVICE)
        edge_type = torch.zeros(size=(max_edges,)).to(DEVICE)
        return (x_indices, edge_index, edge_type)
    edge_sources_indices = list(map(lambda idx: idx + grapharea_size,
                                    [(in_tensor[grapharea_size:grapharea_size + max_edges] != -1).nonzero().flatten()]))
    edge_sources = in_tensor[edge_sources_indices]
    edge_destinations_indices = list(map(lambda idx: idx + grapharea_size + max_edges,
                                         [(in_tensor[
                                           grapharea_size + max_edges:grapharea_size + 2 * max_edges] != -1).nonzero().flatten()]))
    edge_destinations = in_tensor[edge_destinations_indices]
    edge_type_indices = list(map(lambda idx: idx + grapharea_size + 2 * max_edges,
                                 [(in_tensor[grapharea_size + 2 * max_edges:] != -1).nonzero().flatten()]))
    edge_type = in_tensor[edge_type_indices]

    edge_index = torch.stack([edge_sources, edge_destinations], dim=0)

    return (x_indices, edge_index, edge_type)

# splitting into the 2 parts, globals and senses
def unpack_input_tensor(in_tensor, grapharea_size):

    max_edges = int(grapharea_size**1.5)
    if len(in_tensor.shape) > 1:
        in_tensor = in_tensor.squeeze()

    in_tensor_globals, in_tensor_senses = torch.split(in_tensor, split_size_or_sections=in_tensor.shape[0]//2, dim=0)
    (x_indices_g, edge_index_g, edge_type_g) = unpack_to_input_tpl(in_tensor_globals, grapharea_size, max_edges)
    (x_indices_s, edge_index_s, edge_type_s) = unpack_to_input_tpl(in_tensor_senses, grapharea_size, max_edges)
    return ((x_indices_g, edge_index_g, edge_type_g), (x_indices_s, edge_index_s, edge_type_s))

##### 1.4: forward() loop to create the input signal(s), over batch elements
def get_input_signals(model, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls):
    batch_elems_at_t = batch_elements_at_t.squeeze(dim=1)
    elems_at_t_ls = batch_elements_at_t.chunk(chunks=batch_elems_at_t.shape[0], dim=0)

    t_input_lts = [unpack_input_tensor(sample_tensor, model.grapharea_size) for sample_tensor in elems_at_t_ls]
    t_globals_indices_ls = [t_input_lts[b][0][0] for b in range(len(t_input_lts))]

    # -------------------- Input --------------------
    # Input signal n.1: the embedding of the current (global) word
    t_current_globals_indices_ls = [x_indices[0] - model.last_idx_senses for x_indices in t_globals_indices_ls]
    t_current_globals_indices = torch.stack(t_current_globals_indices_ls, dim=0)
    t_word_embeddings = model.E.index_select(dim=0, index=t_current_globals_indices)
    word_embeddings_ls.append(t_word_embeddings)
    # Input signal n.2: the node-state of the current global word - now with graph batching
    if model.include_globalnode_input:
        t_g_nodestates = run_graphnet(t_input_lts, batch_elems_at_t, t_globals_indices_ls, model)
        currentglobal_nodestates_ls.append(t_g_nodestates)


#### 1.4b: Graph Neural Networks for globals
def run_graphnet(t_input_lts, batch_elems_at_t, t_globals_indices_ls, model):

    currentglobal_nodestates_ls = []
    if model.include_globalnode_input:
        t_edgeindex_g_ls = [t_input_lts[b][0][1] for b in range(len(t_input_lts))]
        t_edgetype_g_ls = [t_input_lts[b][0][2] for b in range(len(t_input_lts))]
        for i_sample in range(batch_elems_at_t.shape[0]):
            sample_edge_index = t_edgeindex_g_ls[i_sample]
            sample_edge_type = t_edgetype_g_ls[i_sample]
            x_indices, edge_index, edge_type = lemmatize_node(t_globals_indices_ls[i_sample], sample_edge_index, sample_edge_type, model=model)
            sample_x = model.X.index_select(dim=0, index=x_indices.squeeze())
            x_attention_states = model.gat_globals(sample_x, edge_index)
            currentglobal_node_state = x_attention_states.index_select(dim=0, index=model.select_first_indices[0].to(
                torch.int64))
            currentglobal_nodestates_ls.append(currentglobal_node_state)

        t_currentglobal_node_states = torch.stack(currentglobal_nodestates_ls, dim=0).squeeze(dim=1)
        return t_currentglobal_node_states

##### 1.5: standard LM prediction with globals, using a GRU
def predict_globals_withGRU(model, batch_input_signals, seq_len, distributed_batch_size):
    # ------------------- Globals -------------------
    # - input of shape(seq_len, batch_size, input_size): tensor containing the features of the input sequence.
    task_1_out = rnn_loop(batch_input_signals, model, model.main_rnn_ls, model.memory_hn)  # self.network_1_L1(input)
    task_1_out = task_1_out.permute(1, 0, 2)  # going to: (batch_size, seq_len, n_units)

    task_1_out = task_1_out.reshape(distributed_batch_size * seq_len, task_1_out.shape[2])

    logits_global = model.linear2global(task_1_out)  # shape=torch.Size([5])
    predictions_globals = tfunc.log_softmax(logits_global, dim=1)

    return predictions_globals, logits_global

##### 1.6: Choose the method to compute the location context / Self-attention query
class ContextMethod(Enum):
    AVERAGE = "The context at location t is the average of the last [t-C,...,t] tokens"
    GRU = "The context at location t is the output of a 2-layer GRU"


################################
### 2: Lemmatize global node ###
################################


def lemmatize_node(x_indices, edge_index, edge_type, model):
    currentglobal_relative_X_idx = x_indices[0]
    currentglobal_absolute_vocab_idx = currentglobal_relative_X_idx - model.last_idx_senses
    word = model.vocabulary_wordList[currentglobal_absolute_vocab_idx]
    lemmatized_word = model.vocabulary_lemmatizedList[currentglobal_absolute_vocab_idx]

    logging.debug("***\nword=" + str(word) + " ; lemmatized_word= "+ str(lemmatized_word))

    num_dummy_senses = len(list(filter(lambda n: model.first_idx_dummySenses < n and n < model.last_idx_senses, x_indices)))

    # if a word has edges that are not all self-loops, do not lemmatize it (to avoid turning 'as' into 'a')
    if len(edge_type)>num_dummy_senses:
        logging.debug("word has edges that are not all connections to dummySenses. We don't lemmatize")
        return x_indices, edge_index, edge_type
    if lemmatized_word != word:  # if the lemmatized word is actually different from the original, get the data
        try:
            logging.debug("Getting the data for the lemmatized word")
            lemmatized_word_absolute_idx = model.vocabulary_wordList.index(lemmatized_word)
            lemmatized_word_relative_idx = lemmatized_word_absolute_idx + model.last_idx_senses
            (x_indices_lemmatized, edge_index_lemmatized, edge_type_lemmatized) = \
                AD.get_node_data(model.grapharea_matrix, lemmatized_word_relative_idx, model.grapharea_size)
            return x_indices_lemmatized, edge_index_lemmatized, edge_type_lemmatized
        except ValueError:
            # the lemmatized word was not found in the vocabulary.
            logging.debug("The lemmatized word was not found in the vocabulary")
            return x_indices, edge_index, edge_type
    else:
        return x_indices, edge_index, edge_type

