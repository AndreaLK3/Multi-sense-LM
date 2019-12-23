import torch
from torch_geometric.nn import RGCNConv
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tF
import GraphNN.DefineGraph as DG
import GraphNN.SenseLabeledCorpus as SLC
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
import Vocabulary.Vocabulary_Utilities as VocabUtils
import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
from math import inf
import GraphNN.BatchesRGCN as BatchesRGCN

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



### Internal function to: translate the word (and if present, the sense) into numerical indices.
# sense = [0,se) ; single prototype = [se,se+sp) ; definitions = [se+sp, se+sp+d) ; examples = [se+sp+d, e==num_nodes)
def convert_tokendict_to_tpl(token_dict, senseindices_db_c, globals_vocabulary_h5, last_idx_senses):
    keys = token_dict.keys()
    sense_index_queryresult = None

    if 'wn30_key' in keys:
        try:
            wordnet_sense = wn.lemma_from_key(token_dict['wn30_key']).synset().name()
            logging.debug(wordnet_sense)
            query = "SELECT vocab_index FROM indices_table " + "WHERE word_sense='" + wordnet_sense + "'"
            sense_index_queryresult = senseindices_db_c.execute(query).fetchone()
        except WordNetError: # it may fail, due to typo or wrong labeling
            logging.info("Did not find word sense for key = " + token_dict['wn30_key'])
        except sqlite3.OperationalError :
            logging.info("Error while attempting to execute query: " + query + " . Skipping sense")

        if sense_index_queryresult is None: # the was no sense-key, or we did not find the sense for the key
            sense_index = -1
        else:
            sense_index = sense_index_queryresult[0]
    else:
        sense_index = -1
    word = VocabUtils.process_slc_token(token_dict)
    try:
        global_absolute_index = Utils.select_from_hdf5(globals_vocabulary_h5, 'vocabulary', ['word'], [word]).index[0]
    except IndexError:
        # global_absolute_index = Utils.select_from_hdf5(globals_vocabulary_h5, 'vocabulary', ['word'], [Utils.UNK_TOKEN]).index[0]
        raise Utils.MustSkipUNK_Exception

    global_index = global_absolute_index # + last_idx_senses; do not add this to globals, or we go beyond the n_classes
    logging.debug('(global_index, sense_index)=' + str((global_index, sense_index)))
    return (global_index, sense_index)


### Entry point function to: translate the word (and if present, the sense) into numerical indices.
def get_tokens_tpls(next_token_tpl, split_datagenerator, senseindices_db_c, vocab_h5, last_idx_senses):
    if next_token_tpl is None:
        current_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),
                                                     senseindices_db_c, vocab_h5, last_idx_senses)
        next_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),
                                                  senseindices_db_c, vocab_h5, last_idx_senses)
    else:
        current_token_tpl = next_token_tpl
        next_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),senseindices_db_c, vocab_h5, last_idx_senses)
    return current_token_tpl, next_token_tpl


### Part of an iteration of the training loop. Also defines the graph segment we operate on.
def compute_loss_iteration(data, model, graphbatch_size, current_token_tpl, next_token_tpl):

    (current_input_global, current_input_sense) = current_token_tpl

    if current_input_sense == -1:
        current_token_index = current_input_global
    else:
        current_token_index = current_input_sense
    batch_x, batch_edge_index, batch_edge_type = BatchesRGCN.get_batch_of_graph(current_token_index, graphbatch_size, data)
    predicted_globals, predicted_senses = model(batch_x, batch_edge_index, batch_edge_type)

    logging.debug('current_token_tpl=' + str(current_token_tpl))
    logging.debug('next_token_tpl=' + str(next_token_tpl))
    (y_labelnext_global, y_labelnext_sense) = next_token_tpl

    if y_labelnext_sense == -1:
        is_label_senseLevel = False
        y_labelnext_sense = 0 # it is not used anyway. May be useful to mantain the assertion: labels \in [0, num_classes)
    else:
        is_label_senseLevel = True

    predicted_globals = predicted_globals.unsqueeze(0) # adding 1 dimension, since we do not use batches for now. N x C
    y_labelnext_global = torch.Tensor([y_labelnext_global]).to(torch.int64).to(DEVICE) # N
    logging.debug('y_labelnext_global= ' + str(y_labelnext_global))
    logging.debug('predicted_globals.shape= ' + str(predicted_globals.shape))
    loss_global = tF.nll_loss(predicted_globals, y_labelnext_global)

    if is_label_senseLevel:
        predicted_senses = predicted_senses.unsqueeze(0)
        y_labelnext_sense = torch.Tensor([y_labelnext_sense]).to(torch.int64).to(DEVICE) # N
        loss_sense = tF.nll_loss(predicted_senses, y_labelnext_sense)
        loss = loss_global + loss_sense
    else:
        loss = loss_global



    return loss


def train():
    Utils.init_logging('MyRGCN_train.log')

    data = DG.get_graph_dataobject(new=False)
    model = NetRGCN(data)
    logging.info("Graph-data object loaded, model initialized. Moving them to GPU device(s) if present.")
    data.to(DEVICE), model.to(DEVICE)

    senseindices_db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()

    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCAB_FROMSLC_FILE)
    vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    num_epochs = 10
    model.train()
    losses = []
    graphbatch_size = 16
    steps_logging = 10

    for epoch in range(1,num_epochs+1):
        logging.info("\nTraining epoch n."+str(epoch) +":" )
        train_generator = SLC.read_split('training')
        next_token_tpl = None
        valid_generator = SLC.read_split('validation')
        epoch_valid_loss = inf
        step = 0
        try:
            while(True):
                logging.info("Step n." + str(step))
                try:
                    current_token_tpl, next_token_tpl = get_tokens_tpls(next_token_tpl, train_generator,
                                                                        senseindices_db_c,vocab_h5, model.last_idx_senses)
                except Utils.MustSkipUNK_Exception:
                    logging.info("Encountered <UNK> token. Node not connected to any other in the graph, skipping")
                    continue

                optimizer.zero_grad()
                loss = compute_loss_iteration(data, model, graphbatch_size, current_token_tpl, next_token_tpl)
                loss.backward()
                optimizer.step()
                losses.append(loss)

                step = step +1
                if step % steps_logging == 0:
                    logging.info("Step n." + str(step))
                    logging.info('nll_loss= ' + str(loss))


        except StopIteration:
            continue # next epoch

    getLossGraph(losses)

def getLossGraph(source1):
    plt.plot(source1, color='red', marker='o')
