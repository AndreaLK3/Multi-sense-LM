import torch
from torch_geometric.nn import RGCNConv
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tfunc
import torch.nn.modules.batchnorm as batchnorm
import Graph.DefineGraph as DG
import GNN.SenseLabeledCorpus as SLC
import sqlite3
import os
import pandas as pd
from math import inf
import Graph.Adjacencies as AD
import numpy as np
from time import time
from Utils import DEVICE
import GNN.DataLoading as DL

### The Graph Neural Network. Currently, it has:
###     1 RGCN layer that operates on the selected area of the the graph
###     2 linear layers, that go from the RGCN representation to the global classes and the senses' classes
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

    def forward(self, batchinput_ls):  # given the batches, the current node is at index 0
        predictions_globals_ls = []
        predictions_senses_ls = []
        for (x, edge_index, edge_type) in batchinput_ls:
            rgcn_conv = self.conv1(x, edge_index, edge_type)
            # normalizer = batchnorm.BatchNorm1d(num_features=x.shape[1])
            # normalized_rgcn_conv = normalizer(rgcn_conv)
            x_Lplus1 = tfunc.relu(rgcn_conv)
            x1_current_node = x_Lplus1[0]  # current_node_index
            logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
            logits_sense = self.linear2sense(x1_current_node)

            sample_predictions_globals = tfunc.log_softmax(logits_global, dim=0)
            predictions_globals_ls.append(sample_predictions_globals)
            sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
            predictions_senses_ls.append(sample_predictions_senses)

        return torch.stack(predictions_globals_ls, dim=0), torch.stack(predictions_senses_ls, dim=0)


########

# Auxiliary function for compute_model_loss
def compute_sense_loss(predictions_senses, batch_labels_senses):
    batch_validsenses_predicted = []
    batch_validsenses_labels = []
    for i in range(batch_labels_senses.shape[0]):
        senselabel = batch_labels_senses[i]
        if senselabel != -1:
            batch_validsenses_labels.append(senselabel.item())
            batch_validsenses_predicted.append(predictions_senses[i])
    if len(batch_validsenses_labels) > 1:
        loss_sense = tfunc.nll_loss(torch.stack(batch_validsenses_predicted).to(DEVICE),
                                    torch.tensor(batch_validsenses_labels, dtype=torch.int64).to(DEVICE))
    else:
        loss_sense = 0
    return loss_sense

########

def compute_model_loss(model,batch_input, batch_labels):
    predictions_globals, predictions_senses = model(batch_input)

    batch_labels_t = torch.tensor(batch_labels).t().to(DEVICE)
    batch_labels_globals = batch_labels_t[0]
    batch_labels_senses = batch_labels_t[1]

    # compute the loss for the batch
    loss_global = tfunc.nll_loss(predictions_globals, batch_labels_globals)
    loss_sense = compute_sense_loss(predictions_senses, batch_labels_senses)
    loss = loss_global + loss_sense

    return loss

########

def train(grapharea_size=32, batch_size=8, learning_rate=0.003, num_epochs=100):
    Utils.init_logging('MyRGCN.log')
    graph_dataobj = DG.get_graph_dataobject(new=False)
    logging.info(graph_dataobj)
    model = NetRGCN(graph_dataobj)
    logging.info("Graph-data object loaded, model initialized. Moving them to GPU device(s) if present.")
    graph_dataobj.to(DEVICE)
    model.to(DEVICE)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
         model = torch.nn.DataParallel(model)
         model = model.module

    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, grapharea_size)

    senseindices_db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()

    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE)
    vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #  weight_decay=0.0005

    model.train()
    training_losses_lts = []
    validation_losses_lts = []
    steps_logging = 100 // batch_size
    hyperparams_str = 'batch' + str(batch_size) \
                      + '_area' + str(grapharea_size)\
                      + '_lr' + str(learning_rate) \
                      + '_epochs' + str(num_epochs)
    trainlosses_fpath = os.path.join(F.FOLDER_GRAPHNN, hyperparams_str + '_' + Utils.TRAINING + '_' + F.LOSSES_FILEEND)
    validlosses_fpath = os.path.join(F.FOLDER_GRAPHNN, hyperparams_str + '_' + Utils.VALIDATION + '_' + F.LOSSES_FILEEND)

    global_step = 0
    previous_valid_loss = inf

    for epoch in range(1,num_epochs+1):
        logging.info("\nTraining epoch n."+str(epoch) + ":")
        train_dataset = DL.TextDataset('training', senseindices_db_c, vocab_h5, model,
                                       grapharea_matrix, grapharea_size, graph_dataobj)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                                       collate_fn=DL.collate_fn)

        sum_epoch_loss = 0
        epoch_step = 0

        flag_earlystop = False

        for batch_input, batch_labels in train_dataloader: # tuple of 2 lists
            # starting operations on one batch
            optimizer.zero_grad()
            t0 = time()
            # compute loss for the batch
            loss = compute_model_loss(model, batch_input, batch_labels)
            # running sum of the training loss in the log segment
            sum_epoch_loss = sum_epoch_loss + loss.item()

            loss.backward()
            optimizer.step()

            global_step = global_step + 1
            epoch_step = epoch_step + 1
            if global_step % steps_logging == 0:
                logging.info("Global step=" + str(global_step) + "\t ; Iteration time=" + str(round(time()-t0,5)))

            #Utils.log_chronometer([t0, time()])
            #logging.info('Epoch step n.' + str(epoch_step) + ", using batch_size=" + str(batch_size))


        # except StopIteration: the DataLoader naturally catches StopIteration
            # end of an epoch.
        logging.info("-----\n End of epoch. Global step n." + str(global_step) + ", using batch_size=" + str(batch_size))
        epoch_loss = sum_epoch_loss / epoch_step
        logging.info("Training, epoch nll_loss= " + str(round(epoch_loss,5)) + '\n------')
        training_losses_lts.append(epoch_loss) # (num_epochs, epoch_loss)
        # Time to check the validation loss
        # early stopping disabled for the test of overfitting on small dataset
        continue
        valid_dataset = DL.TextDataset('validation', senseindices_db_c, vocab_h5, model,
                                       grapharea_matrix, grapharea_size, graph_dataobj)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=1,
                                                       collate_fn=DL.collate_fn)
        epoch_valid_loss = validation(valid_dataloader, model)

        if epoch_valid_loss > previous_valid_loss + 0.01:  # ( + epsilon)
            flag_earlystop = True
        previous_valid_loss = epoch_valid_loss

        if flag_earlystop:
            break

    torch.save(graph_dataobj, os.path.join(F.FOLDER_GRAPHNN, hyperparams_str +
                                  'step_' + str(global_step) + '_graph.dataobject'))
    torch.save(model,os.path.join(F.FOLDER_GRAPHNN, hyperparams_str +
                                  'step_' + str(global_step) + '.rgcnmodel'))

    np.save(trainlosses_fpath, np.array(training_losses_lts))
    np.save(validlosses_fpath, np.array(validation_losses_lts))



#####

def validation(valid_dataloader, model):

    model.eval()  # do not train the model now
    sum_epoch_valid_loss = 0
    validation_step = 0
    logging_step = 1000

    with torch.no_grad: # Deactivates the autograd engine entirely to save some memory
        for batch_input, batch_labels in valid_dataloader:
            valid_loss = compute_model_loss(model, batch_input, batch_labels)
            sum_epoch_valid_loss = sum_epoch_valid_loss + valid_loss
            validation_step = validation_step + 1
            if validation_step % logging_step == 0:
                logging.info("Validation step n. " + str(validation_step))

    epoch_valid_loss = sum_epoch_valid_loss / validation_step

    model.train()  # training can resume

    return epoch_valid_loss



