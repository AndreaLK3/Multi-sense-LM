import logging

import torch
from torch.nn import functional as tF

import Utils
from GNN import NumericalIndices as IN
from Graph import Adjacencies as AD
from Utils import DEVICE

# ### Auxiliary function for step n.2, to Pad the tensors of the input elements x, edge_index and edge_type
# ### (that for instance can have shapes [<=32, 300], [2,2048] and [2048] respectively)
# ### We pad, and then stack side-by-side, to obtain a matrix that can be stacked with the other elements of the batch.
# def get_forwardinput_matrix_forelement(area_x, edge_index, edge_type, grapharea_size):
#     M_x = torch.ones(size=(grapharea_size, area_x.shape[1])) * -1
#     M_x[0:area_x.shape[0], 0:area_x.shape[1]] = area_x
#
#     M_edge_index = torch.ones(size=(grapharea_size, edge_index.shape[1])) * -1
#     M_edge_index[0:edge_index.shape[0], 0:edge_index.shape[1]] = edge_index
#
#     M_edge_type = torch.ones(size=(grapharea_size, edge_type.shape[0])) * -1
#     M_edge_type[0:edge_type.shape[0], 0:edge_type.shape[0]] = edge_type
#
#     M = torch.cat([M_x, M_edge_index, M_edge_type], dim=1)
#     return M
#

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





