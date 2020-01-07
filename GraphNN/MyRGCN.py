import torch
from torch_geometric.nn import RGCNConv
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tF
import GraphNN.DefineGraph as DG
import GraphNN.SenseLabeledCorpus as SLC
import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
from math import inf
import GraphNN.NumericalIndices as IN
import GraphNN.GraphArea as GraphArea
import numpy as np
from time import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NetRGCN(torch.nn.Module):
    def __init__(self, data):
        super(NetRGCN, self).__init__()
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.conv1 = RGCNConv(in_channels=data.x.shape[1], # doc: "Size of each input sample " in the example, num_nodes
                              out_channels=data.x.shape[1], # doc: "Size of each output sample "
                              num_relations=data.num_relations,
                              num_bases=data.num_relations)
        self.linear2global = torch.nn.Linear(in_features=data.x.shape[1],
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        self.linear2sense = torch.nn.Linear(in_features=data.x.shape[1], out_features=self.last_idx_senses, bias=True)


    def forward(self, batch_x, batch_edge_index, batch_edge_type):  # given the batches, the current node is at index 0
        x_Lplus1 = tF.relu(self.conv1(batch_x, batch_edge_index, batch_edge_type))
        x1_current_node = x_Lplus1[0]  # current_node_index
        logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
        logits_sense = self.linear2sense(x1_current_node)

        return (tF.log_softmax(logits_global, dim=0), tF.log_softmax(logits_sense, dim=0))


# used to compute the loss, among the elements in a batch, for the 2 categories: globals and senses
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

# Given the tuple of numerical indices for an element, e.g. (13,5), retrieve a list
# of either 1 or 2 tuples that are input to the forward function, i.e. (x, edge_index, edge_type)
# Here Is decide what is the starting token for a prediction. For now, it is "sense if present, else global"
def get_forwardinput_forelement(global_idx, sense_idx, node_area_size, graph):
    forward_input_ls = []
    logging.debug("get_forwardinput_forelement: " + str(global_idx) + ' ; ' + str(sense_idx))
    if (sense_idx == -1):
        x, edge_index, edge_type = GraphArea.get_graph_area(global_idx, node_area_size, graph)
        forward_input_ls.append((x, edge_index, edge_type))
    else:
        x, edge_index, edge_type = GraphArea.get_graph_area(sense_idx, node_area_size, graph)
        forward_input_ls.append((x, edge_index, edge_type))
    return forward_input_ls


def compute_validation_loss(model, valid_generator, senseindices_db_c, vocab_h5, grapharea_size, data):
    model.eval()
    validation_losses_ls = []
    validation_batch_size = 64 # by default, losses are averaged in a batch. I introduce batches to be faster
    try:
        while(True):
            with torch.no_grad():
                batch_valid_loss = select_and_process_batch(validation_batch_size, grapharea_size,
                                                            valid_generator, senseindices_db_c, vocab_h5, model, data)
                validation_losses_ls.append(batch_valid_loss)
    except StopIteration:
        pass # iterating over the validation split finished

    valid_loss = np.average(validation_losses_ls)
    logging.info("Validation loss=" + valid_loss)
    model.train()
    return valid_loss


def select_and_process_batch(batch_size, grapharea_size, elements_generator, senseindices_db_c, vocab_h5, model, data):
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

    # gathering the graph-input for the RGCN layer
    batch_rgcn_input_ls = []
    for i in range(len(input_indices_lts) - 1):
        (global_idx, sense_idx) = input_indices_lts[i]
        batch_rgcn_input_ls.extend(
            get_forwardinput_forelement(global_idx, sense_idx, grapharea_size, data))

    # computing the loss for the batch
    loss = compute_batch_losses(input_indices_lts, batch_rgcn_input_ls, model)
    return loss


def train(grapharea_size=32, batch_size=8, num_epochs=50):
    Utils.init_logging('temp.log')
    data = DG.get_graph_dataobject(new=False)
    model = NetRGCN(data)
    logging.info("Graph-data object loaded, model initialized. Moving them to GPU device(s) if present.")
    data.to(DEVICE), model.to(DEVICE)

    senseindices_db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()

    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE)
    vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    model.train()
    losses_lts = []
    perplexity_values_lts = []
    validation_losses_lts = []
    steps_logging = 100 // batch_size
    hyperparameters_fnamestring = 'batchsize' + str(batch_size) \
                                  + '_graphareasize' + str(grapharea_size)\
                                  + '_epochs' + str(num_epochs)
    trainlosses_record_fpath = os.path.join(F.FOLDER_GRAPHNN, hyperparameters_fnamestring + '_' + F.LOSSES_FILEEND)
    perplexity_record_fpath = os.path.join(F.FOLDER_GRAPHNN, hyperparameters_fnamestring + '_' + F.PERPLEXITY_FILEEND)

    for epoch in range(1,num_epochs+1):
        logging.info("\nTraining epoch n."+str(epoch) + ":")
        train_generator = SLC.read_split('training')

        valid_generator = SLC.read_split('validation')
        previous_valid_loss = inf
        step = 0

        try:
            while(True):

                # starting operating on one batch
                optimizer.zero_grad()
                loss = select_and_process_batch(batch_size, grapharea_size, train_generator, senseindices_db_c, vocab_h5, model, data)
                loss.backward()
                optimizer.step()

                step = step +1
                if step % steps_logging == 0:
                    logging_point = epoch * step // steps_logging
                    logging.info("Logging point n." + str(logging_point) + ';' + 'Training nll_loss= ' + str(loss))
                    losses_lts.append((logging_point,loss.item()))
                    # calculating perplexity=e^(cross_entropy). We currently have log_softmax>nll_loss, equivalent to it
                    perplexity = torch.exp(loss)
                    perplexity_values_lts.append((logging_point,perplexity))

        except StopIteration:
            epoch_valid_loss = compute_validation_loss(model, valid_generator, senseindices_db_c, vocab_h5)
            validation_logging_point = epoch * step // steps_logging
            validation_losses_lts.append((validation_logging_point,epoch_valid_loss))
            if epoch_valid_loss > previous_valid_loss + 0.01: # (epsilon)
                torch.save(data, os.path.join(F.FOLDER_GRAPHNN, hyperparameters_fnamestring + '.graphdata'))
                torch.save(model, os.path.join(F.FOLDER_GRAPHNN, hyperparameters_fnamestring + '.rgcnmodel'))
                break
            else:
                continue # next epoch

    np.save(trainlosses_record_fpath, np.array(losses_lts))
    np.save(perplexity_record_fpath, np.array(losses_lts))