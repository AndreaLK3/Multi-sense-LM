import torch
from Utils import DEVICE
import Graph.Adjacencies as AD

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

#### Graph Neural Networks on global nodes (input signal n.2)
def run_graphnet(t_input_lts, batch_elems_at_t, t_globals_indices_ls, model):

    currentglobal_nodestates_ls = []
    if model.include_globalnode_input > 0:
        t_edgeindex_g_ls = [t_input_lts[b][0][1] for b in range(len(t_input_lts))]
        t_edgetype_g_ls = [t_input_lts[b][0][2] for b in range(len(t_input_lts))]
        for i_sample in range(batch_elems_at_t.shape[0]):
            sample_edge_index = t_edgeindex_g_ls[i_sample]
            sample_edge_type = t_edgetype_g_ls[i_sample]
            x_indices, edge_index, edge_type = AD.lemmatize_node(t_globals_indices_ls[i_sample], sample_edge_index, sample_edge_type, model=model)
            sample_x = model.X.index_select(dim=0, index=x_indices.squeeze())
            x_attention_states = model.gat_globals(sample_x, edge_index)
            currentglobal_node_state = x_attention_states.index_select(dim=0, index=model.select_first_indices[0].to(
                torch.int64))
            currentglobal_nodestates_ls.append(currentglobal_node_state)

        t_currentglobal_node_states = torch.stack(currentglobal_nodestates_ls, dim=0).squeeze(dim=1)
        return t_currentglobal_node_states

# Starting from the batch, collect the input signals: word embeddings, and possibly graph node embeddings
def get_input_signals(model, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls,
                      input_indices_ls=None):
    if model.__class__.__name__.lower() != "standardlm":
        model = model.StandardLM

    batch_elems_at_t = batch_elements_at_t.squeeze(dim=1)
    elems_at_t_ls = batch_elements_at_t.chunk(chunks=batch_elems_at_t.shape[0], dim=0)

    t_input_lts = [unpack_input_tensor(sample_tensor, model.grapharea_size) for sample_tensor in elems_at_t_ls]
    t_globals_indices_ls = [t_input_lts[b][0][0] for b in range(len(t_input_lts))]

    # -------------------- Input --------------------
    # Input signal n.1: the embedding of the current word
    t_current_globals_indices_ls = [x_indices[0] - model.last_idx_senses for x_indices in t_globals_indices_ls]
    t_current_globals_indices = torch.stack(t_current_globals_indices_ls, dim=0)
    if input_indices_ls is not None:
        input_indices_ls.append(t_current_globals_indices)
    t_word_embeddings = model.E.index_select(dim=0, index=t_current_globals_indices)
    word_embeddings_ls.append(t_word_embeddings)
    # Input signal n.2: the node-state of the current word (i.e. its global node) - now with graph batching
    if model.include_globalnode_input > 0:
        t_g_nodestates = run_graphnet(t_input_lts, batch_elems_at_t, t_globals_indices_ls, model)
        currentglobal_nodestates_ls.append(t_g_nodestates)