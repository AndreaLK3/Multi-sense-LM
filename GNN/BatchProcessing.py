import logging

import torch
from torch.nn import functional as tF

import Utils
from GNN import NumericalIndices as IN
from Graph import Adjacencies as SA
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


### Batch step n.2: given the numerical indices of the words,
### invoke the function that gathers the graph-input for each node.
### Graph-input= neigbouring nodes in x, their edges in edge_index, edge_type
def get_batch_grapharea_input(input_indices_lts, grapharea_matrix, area_size):
    # gathering the graph-input for the RGCN layer
    batch_rgcn_input_ls = []
    for i in range(len(input_indices_lts) - 1):
        (global_idx, sense_idx) = input_indices_lts[i]

        batch_rgcn_input_ls.extend(
            get_forwardinput_forelement(global_idx, sense_idx, grapharea_matrix, area_size))
    return batch_rgcn_input_ls

### Auxiliary function, to get the graph-input (x, edge_index, edge_type)
def get_forwardinput_forelement(global_idx, sense_idx, grapharea_matrix, area_size, graph_dataobj):
    forward_input_ls = []
    logging.info("get_forwardinput_forelement: " + str(global_idx) + ' ; ' + str(sense_idx))
    if (sense_idx == -1):
        nodes_ls, edge_index, edge_type = SA.get_node_data(grapharea_matrix, global_idx, area_size)
        area_x = graph_dataobj.x.index_select(0, nodes_ls)
        # old version: x, edge_index, edge_type = GraphArea.get_graph_area(global_idx, node_area_size, graph)
        forward_input_ls.append((area_x, edge_index, edge_type))
    else:
        nodes_ls, edge_index, edge_type = SA.get_node_data(grapharea_matrix, sense_idx, area_size)
        area_x = graph_dataobj.x.index_select(0, nodes_ls)
        forward_input_ls.append((area_x, edge_index, edge_type))
    return forward_input_ls


### Batch step n.4: compute softmax>nll_loss (a.k.a Cross-Entropy) loss
def compute_batch_losses(input_indices_lts, batch_rgcn_input_ls, model):
    # predictions and labels. The sense prediction is included only if the sense label is valid.
    batch_predicted_globals_ls = []
    batch_predicted_senses_ls = []
    batch_labels_globals_ls = []
    batch_labels_senses_ls = []

    for i in range(len(input_indices_lts) - 1):
        (x, edge_index, edge_type) = batch_rgcn_input_ls[i]

        predicted_globals, predicted_senses = model(x, edge_index, edge_type)

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





