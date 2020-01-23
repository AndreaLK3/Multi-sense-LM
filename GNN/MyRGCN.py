import torch
from torch_geometric.nn import RGCNConv
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tF
import Graph.DefineGraph as DG
import GNN.SenseLabeledCorpus as SLC
import sqlite3
import os
import pandas as pd
from math import inf
import Graph.Adjacencies as SA
import numpy as np
from time import time
import GNN.BatchProcessing as BP
from Utils import DEVICE


### The Graph Neural Network. Currently, it has:
###     1 RGCN layer that operates on the selected area of the the graph
###     2 linear layers, that go from the RGCN representation to the  global classes and the senses' classes
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
        logging.info("NetRGCN > forward")
        logging.info("batch_x = " + str(batch_x))
        logging.info("batch_edge_index = " + str(batch_edge_index))
        logging.info("batch_edge_type = " + str(batch_edge_type))
        x_Lplus1 = tF.relu(self.conv1(batch_x, batch_edge_index, batch_edge_type))
        x1_current_node = x_Lplus1[0]  # current_node_index
        logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
        logits_sense = self.linear2sense(x1_current_node)

        return (tF.log_softmax(logits_global, dim=0), tF.log_softmax(logits_sense, dim=0))


# Used to compute the loss, among the elements in a batch, for the 2 categories: globals and senses

# Given the tuple of numerical indices for an element, e.g. (13,5), retrieve a list
# of either 1 or 2 tuples that are input to the forward function, i.e. (x, edge_index, edge_type)
# Here Is decide what is the starting token for a prediction. For now, it is "sense if present, else global"


def compute_validation_loss(model, valid_generator, senseindices_db_c, vocab_h5, grapharea_matrix, grapharea_size):
    model.eval()
    validation_losses_ls = []
    validation_batch_size = 64 # by default, losses are averaged in a batch. I introduce batches to be faster
    try:
        while(True):
            with torch.no_grad():
                input_indices_lts = BP.select_batch_indices(validation_batch_size, valid_generator,senseindices_db_c, vocab_h5, model)
                batch_grapharea_input = BP.get_batch_grapharea_input(input_indices_lts, grapharea_matrix, grapharea_size)
                batch_valid_loss = BP.compute_batch_losses(input_indices_lts, batch_grapharea_input, model)
                validation_losses_ls.append(batch_valid_loss.item())
    except StopIteration:
        pass # iterating over the validation split finished

    valid_loss = np.average(validation_losses_ls)
    logging.info("Validation loss=" + str(round(valid_loss, 5)))
    model.train()
    return valid_loss


def train(grapharea_size=32, batch_size=1, num_epochs=5):
    Utils.init_logging('MyRGCN.log')
    graph_dataobj = DG.get_graph_dataobject(new=False)
    logging.info(graph_dataobj)
    model = NetRGCN(graph_dataobj)
    logging.info("Graph-data object loaded, model initialized. Moving them to GPU device(s) if present.")
    graph_dataobj.to(DEVICE)
    model.to(DEVICE)


    n_gpu = torch.cuda.device_count()
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    #     model = model.module

    grapharea_matrix = SA.get_grapharea_matrix(graph_dataobj, grapharea_size)

    senseindices_db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()

    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE)
    vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    model.train()
    losses_lts = []
    validation_losses_lts = []
    steps_logging = 2000 // batch_size
    hyperparams_str = 'batchsize' + str(batch_size) \
                                  + '_graphareasize' + str(grapharea_size)\
                                  + '_epochs' + str(num_epochs)
    trainlosses_fpath = os.path.join(F.FOLDER_GRAPHNN, hyperparams_str + '_' + Utils.TRAINING + '_' + F.LOSSES_FILEEND)
    validlosses_fpath = os.path.join(F.FOLDER_GRAPHNN, hyperparams_str + '_' + Utils.VALIDATION + '_' + F.LOSSES_FILEEND)

    global_step = 0
    for epoch in range(1,num_epochs+1):
        logging.info("\nTraining epoch n."+str(epoch) + ":")
        train_generator = SLC.read_split('training')

        previous_valid_loss = inf
        flag_earlystop = False
        try:
            while(not(flag_earlystop)):
                # starting operating on one batch
                optimizer.zero_grad()
                t0 = time()

                input_indices_lts = BP.select_batch_indices(batch_size, train_generator, senseindices_db_c,
                                                            vocab_h5, model)
                batch_grapharea_input = BP.get_batch_grapharea_input(input_indices_lts, grapharea_matrix,
                                                                     grapharea_size)

                loss = BP.compute_batch_losses(input_indices_lts, batch_grapharea_input, model)
                loss.backward()
                optimizer.step()

                global_step = global_step +1
                if global_step % steps_logging == 0:
                    logging_point = global_step // steps_logging
                    logging.info("Logging point n." + str(logging_point) +
                                 " ; Global step=" + str(global_step) + ' ; Training nll_loss= ' + str(loss))
                    losses_lts.append((logging_point,loss.item()))
                    valid_generator = SLC.read_split('validation')
                    epoch_valid_loss = compute_validation_loss(model, valid_generator, senseindices_db_c, vocab_h5,
                                                               grapharea_size, graph_dataobj)
                    validation_logging_point = global_step // steps_logging
                    validation_losses_lts.append((validation_logging_point, epoch_valid_loss.item()))
                    if epoch_valid_loss > previous_valid_loss + 0.01:  # (epsilon)
                        flag_earlystop = True
                t1 = time()
                Utils.log_chronometer([t0,t1])

        except StopIteration:
            if flag_earlystop:
                torch.save(graph_dataobj, os.path.join(F.FOLDER_GRAPHNN, hyperparams_str +
                                              'step_' + str(global_step) + '_graph.dataobject'))
                torch.save(model,os.path.join(F.FOLDER_GRAPHNN, hyperparams_str +
                                              'step_' + str(global_step) + '.rgcnmodel'))
                break
            else:
                continue # next epoch

    np.save(trainlosses_fpath, np.array(losses_lts))
    np.save(validlosses_fpath, np.array(validation_losses_lts))