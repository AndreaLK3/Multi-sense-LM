import logging

import torch
from torch.nn import functional as tF

import Utils
from GNN import NumericalIndices as IN
from Graph import Adjacencies as AD
from Utils import DEVICE

### Batch step n.1: Use the generator to get the the word tokens, and then convert them into numerical indices
def select_batch_indices(batch_size, elements_generator,senseindices_db_c, vocab_h5, model):
    next_token_tpl = None
    # collecting the numerical indices of the batch's elements
    input_indices_lts = []

    while (len(input_indices_lts) <= batch_size):
        try:
            current_token_tpl, next_token_tpl = IN.get_tokens_tpls(next_token_tpl, elements_generator,
                                                                   senseindices_db_c, vocab_h5,
                                                                   model.last_idx_senses)
            input_indices_lts.append(current_token_tpl)
        except Utils.MustSkipUNK_Exception:
            logging.debug("Encountered <UNK> token. Node not connected to any other in the graph, skipping")
            continue
    logging.debug('input_indices_lts=' + str(input_indices_lts))
    return input_indices_lts

#####################

### Batch step n.2:
### Given the tuple of numerical indices for an element, e.g. (13,5), retrieve the input of the forward function,
### i.e. (x, edge_index, edge_type).
def get_batch_grapharea_input(input_indices_lts, grapharea_matrix, area_size, graph_dataobj):
    # gathering the graph-input for the RGCN layer
    batch_rgcn_input = torch.Tensor([])
    logging.info(input_indices_lts)
    for i in range(len(input_indices_lts) - 1):
        (global_idx, sense_idx) = input_indices_lts[i]
        (area_x, edge_index, edge_type) = get_forwardinput_forelement(global_idx, sense_idx, grapharea_matrix, area_size, graph_dataobj)
        elem_forwardinput_matrix = get_forwardinput_matrix_forelement(area_x, edge_index, edge_type, area_size)

        batch_rgcn_input = torch.cat([batch_rgcn_input, elem_forwardinput_matrix.unsqueeze(0)])

    logging.info(batch_rgcn_input.shape)

    return batch_rgcn_input

### Auxiliary function for step n.2, to get the graph-input (x, edge_index, edge_type)
### Here I decide what is the starting token for a prediction. For now, it is "sense if present, else global"
def get_forwardinput_forelement(global_idx, sense_idx, grapharea_matrix, area_size, graph_dataobj):

    logging.debug("get_forwardinput_forelement: " + str(global_idx) + ' ; ' + str(sense_idx))
    if (sense_idx == -1):
        sourcenode_idx = global_idx
    else:
        sourcenode_idx = sense_idx
    nodes_ls, edge_index, edge_type = AD.get_node_data(grapharea_matrix, sourcenode_idx, area_size)
    valid_nodes_ls = [node_index for node_index in nodes_ls if node_index != -1]
    node_indices = torch.Tensor(sorted(valid_nodes_ls)).to(torch.int64).to(DEVICE)
    area_x = graph_dataobj.x.index_select(0, node_indices)

    return (area_x, edge_index, edge_type)


### Auxiliary function for step n.2, to Pad the tensors of the input elements x, edge_index and edge_type
### (that for instance can have shapes [<=32, 300], [2,2048] and [2048] respectively)
### We pad, and then stack side-by-side, to obtain a matrix that can be stacked with the other elements of the batch.
def get_forwardinput_matrix_forelement(area_x, edge_index, edge_type, grapharea_size):
    M_x = torch.ones(size=(grapharea_size, area_x.shape[1])) * -1
    M_x[0:area_x.shape[0], 0:area_x.shape[1]] = area_x

    M_edge_index = torch.ones(size=(grapharea_size, edge_index.shape[1])) * -1
    M_edge_index[0:edge_index.shape[0], 0:edge_index.shape[1]] = edge_index

    M_edge_type = torch.ones(size=(grapharea_size, edge_type.shape[0])) * -1
    M_edge_type[0:edge_type.shape[0], 0:edge_type.shape[0]] = edge_type

    M = torch.cat([M_x, M_edge_index, M_edge_type], dim=1)
    return M


#####################

### Batch step n.3: compute softmax>nll_loss (a.k.a Cross-Entropy) loss
def compute_batch_losses(input_indices_lts, batch_rgcn_input_mat, model):
    # predictions and labels. The sense prediction is included only if the sense label is valid.
    batch_predicted_globals_ls = []
    batch_predicted_senses_ls = []
    batch_labels_globals_ls = []
    batch_labels_senses_ls = []

    predicted_globals, predicted_senses = model(batch_rgcn_input_mat)

    logging.info(predicted_globals)
    logging.info(predicted_senses)
    raise Exception

    (y_global_idx, y_sense_idx) = input_indices_lts[i + 1]
    (y_global, y_sense) = (
        torch.Tensor([y_global_idx]).type(torch.int64).to(DEVICE),
        torch.Tensor([y_sense_idx]).type(torch.int64).to(DEVICE))
    logging.debug("compute_batch_losses > (y_global, y_sense)=" + str((y_global, y_sense)))

    batch_predicted_globals_ls.extend([predicted_globals])
    batch_labels_globals_ls.extend([y_global])
    if not (y_sense == -1):
        batch_predicted_senses_ls.append(predicted_senses)
        batch_labels_senses_ls.append(y_sense)

    batch_predicted_globals = torch.cat([torch.unsqueeze(t, dim=0) for t in batch_predicted_globals_ls], dim=0)
    batch_labels_globals = torch.cat(batch_labels_globals_ls, dim=0)

    # compute the loss (batch mode)
    loss_global = tF.nll_loss(batch_predicted_globals, batch_labels_globals)
    if len(batch_labels_senses_ls) > 0:
        batch_predicted_senses = torch.cat([torch.unsqueeze(t, dim=0) for t in batch_predicted_senses_ls],
                                           dim=0)
        batch_labels_senses = torch.cat(batch_labels_senses_ls, dim=0)
        loss_sense = tF.nll_loss(batch_predicted_senses, batch_labels_senses)
    else:
        loss_sense = 0
    loss = loss_global + loss_sense

    return loss





