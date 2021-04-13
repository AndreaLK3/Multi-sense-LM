from enum import Enum
import torch
from torch.nn import functional as tfunc
from torch.nn.parameter import Parameter
from Graph.Adjacencies import lemmatize_node
from Utils import DEVICE
import Utils
from torch_geometric.nn import GATConv
from Models.Variants.RNNSteps import rnn_loop

##### Initialization of: graph_dataobj, grapharea_matrix, vocabulary_lists & more
def init_model_parameters(model, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, batch_size, hidden_layers, hidden_units):
    model.grapharea_matrix = grapharea_matrix
    model.grapharea_size = grapharea_size

    model.vocabulary_df = vocabulary_df
    model.vocabulary_wordList = vocabulary_df['word'].to_list().copy()
    model.vocabulary_lemmatizedList = vocabulary_df['lemmatized_form'].to_list().copy()

    model.predict_senses = False # it can be set to True when starting a training loop

    model.first_idx_dummySenses = Utils.compute_startpoint_dummySenses(graph_dataobj) # used to lemmatizeNode in GNNs
    model.last_idx_senses = graph_dataobj.node_types.tolist().index(1)
    model.last_idx_globals = graph_dataobj.node_types.tolist().index(2)

    # These are used for the Senses' GRU
    model.batch_size = batch_size
    model.hidden_layers = hidden_layers
    model.hidden_units = hidden_units

    return


##### Extracting the input elements (x_indices, edge_index, edge_type) from the padded tensor in the batch
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

#### 1.3: Graph Neural Networks for globals
def run_graphnet(t_input_lts, batch_elems_at_t, t_globals_indices_ls, model):

    currentglobal_nodestates_ls = []
    if model.include_globalnode_input > 0:
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

# Starting from the batch, collect the input signals: word embeddings, and possibly graph node embeddings
def get_input_signals(model, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls):
    batch_elems_at_t = batch_elements_at_t.squeeze(dim=1)
    elems_at_t_ls = batch_elements_at_t.chunk(chunks=batch_elems_at_t.shape[0], dim=0)

    t_input_lts = [unpack_input_tensor(sample_tensor, model.grapharea_size) for sample_tensor in elems_at_t_ls]
    t_globals_indices_ls = [t_input_lts[b][0][0] for b in range(len(t_input_lts))]

    # -------------------- Input --------------------
    # Input signal n.1: the embedding of the current word
    if model.include_globalnode_input < 2:
        t_current_globals_indices_ls = [x_indices[0] - model.last_idx_senses for x_indices in t_globals_indices_ls]
        t_current_globals_indices = torch.stack(t_current_globals_indices_ls, dim=0)
        t_word_embeddings = model.E.index_select(dim=0, index=t_current_globals_indices)
        word_embeddings_ls.append(t_word_embeddings)
    # Input signal n.2: the node-state of the current word (i.e. its global node) - now with graph batching
    if model.include_globalnode_input > 0:
        t_g_nodestates = run_graphnet(t_input_lts, batch_elems_at_t, t_globals_indices_ls, model)
        currentglobal_nodestates_ls.append(t_g_nodestates)

##### Choose the method to compute the location context / Self-attention query
class ContextMethod(Enum):
    AVERAGE = "The context at location t is the average of the last [t-C,...,t] tokens"
    GRU = "The context at location t is the output of a 2-layer GRU"


##### Assign =1 to a specific global or sense in the probability distribution #####
# ---------------
def assign_one(target_elements, seq_len, distributed_batch_size, len_output_distr, current_device):

    # ----- preparing the base for the artificial softmax -----
    softmax_distribution = torch.ones((seq_len, distributed_batch_size, len_output_distr)).to(current_device)
    epsilon = 10 ** (-8)
    softmax_distribution = epsilon * softmax_distribution  # base probability value for non-selected senses
    mask = torch.zeros(size=(seq_len, distributed_batch_size, len_output_distr)).to(torch.bool).to(current_device)

    targets = target_elements.clone().reshape((seq_len, distributed_batch_size)).to(torch.torch.int64).to(
        current_device)
    # ----- writing in the artificial softmax -----
    for t in (range(seq_len)):
        for b in range(distributed_batch_size):
            mask[t, b, targets[t, b]] = True
    assign_one = (torch.ones(
        size=mask[mask == True].shape)).to(current_device)
    softmax_distribution.masked_scatter_(mask=mask.data.clone(), source=assign_one)

    predictions = torch.log(softmax_distribution).reshape(seq_len * distributed_batch_size,
                                                           softmax_distribution.shape[2])
    return predictions


