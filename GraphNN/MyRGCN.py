import torch
from torch_geometric.nn import RGCNConv
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tF
import GraphNN.DefineGraph as DG
import GraphNN.SenseLabeledCorpus as SLC
from nltk.corpus import wordnet as wn
import Vocabulary.Vocabulary_Utilities as VocabUtils
import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
from math import inf

class NetRGCN(torch.nn.Module):
    def __init__(self, data):
        super(NetRGCN, self).__init__()
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.conv1 = RGCNConv(in_channels=data.x.shape[1], # doc: "Size of each input sample " in the example, num_nodes
                              out_channels=data.x.shape[1], # doc: "Size of each output sample "
                              num_relations=data.num_relations,
                              num_bases=data.num_relations)
        self.conv2global = RGCNConv(in_channels=data.x.shape[1], out_channels=self.last_idx_globals - self.last_idx_senses,
                              num_relations=data.num_relations, num_bases=data.num_relations)
        self.conv2sense = RGCNConv(in_channels=data.x.shape[1],
                                    out_channels=self.last_idx_senses,
                                    num_relations=data.num_relations, num_bases=data.num_relations)

    def forward(self, x, edge_index, edge_type):
        x_Lplus1 = tF.relu(self.conv1(x, edge_index, edge_type))
        x_toglobal = self.conv2global(x_Lplus1, edge_index, edge_type)
        x_tosense = self.conv2sense(x_Lplus1, edge_index, edge_type)

        # output n.1 : shape [N,G]: for every node, the probability to belong to each one of the Global classes
        # i.e. to be followed by each one of the global words.
        # using softmax on dimension=1 gives sensible probabilities

        return (tF.log_softmax(x_toglobal, dim=1), tF.log_softmax(x_tosense, dim=1))



# sense = [0,se) ; single prototype = [se,se+sp) ; definitions = [se+sp, se+sp+d) ; examples = [se+sp+d, e==num_nodes)

def convert_tokendict_to_tpl(token_dict, senseindices_db_c, globals_vocabulary_h5, last_idx_senses):
    keys = token_dict.keys()

    if 'wn30_key' in keys:
        wordnet_sense = wn.lemma_from_key(token_dict['wn30_key'])
        sense_index = senseindices_db_c.execute("SELECT vocab_index FROM indices_table "+
                                                "WHERE word_sense=="+wordnet_sense)
    else:
        sense_index = -1
    word = VocabUtils.process_slc_token(token_dict)
    global_absolute_index = Utils.select_from_hdf5(globals_vocabulary_h5, 'vocabulary', ['word'], [word]).index[0]
    global_index = global_absolute_index + last_idx_senses
    return (global_index, sense_index)




def compute_loss_iteration(data, model, current_token_tpl, next_token_tpl):

    predicted_global_forEachNode, predicted_sense_forEachNode = model(data.x, data.edge_index, data.edge_type)

    (current_input_global, current_input_sense) = current_token_tpl
    if current_input_sense == -1:
        current_token_index = current_input_global
    else:
        current_token_index = current_input_sense

    (y_labelnext_global, y_labelnext_sense) = next_token_tpl  # (torch.Tensor().type(torch.int64),

    if y_labelnext_sense == -1:
        is_label_senseLevel = False
    else:
        is_label_senseLevel = True

    loss_global = tF.nll_loss(predicted_global_forEachNode[current_token_index].unsqueeze(0), y_labelnext_global)

    if is_label_senseLevel:
        loss_sense = tF.nll_loss(predicted_sense_forEachNode[current_token_index].unsqueeze(0),
                                 y_labelnext_sense)
    else:
        loss_sense = 0

    loss = loss_global + loss_sense
    return loss


def get_tokens_tpls(next_token_tpl, split_datagenerator, senseindices_db_c, vocab_h5, model):
    if next_token_tpl is None:
        current_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),
                                                     senseindices_db_c, vocab_h5, model.last_idx_senses)
        next_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),
                                                  senseindices_db_c, vocab_h5, model.last_idx_senses)
    else:
        current_token_tpl = next_token_tpl
        next_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),
                                                  senseindices_db_c, vocab_h5, model.last_idx_senses)
    return (current_token_tpl, next_token_tpl)


def train():
    Utils.init_logging('temp.log')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputgraph_dataobject = DG.get_graph_dataobject()
    RGCN_modelobject = NetRGCN(inputgraph_dataobject)
    data, model = inputgraph_dataobject.to(device), RGCN_modelobject.to(device)

    senseindices_db_filepath = os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()

    globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCAB_FROMSLC_FILE)
    vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    num_epochs = 10
    model.train()
    losses = []
    batch_size = 1

    for epoch in range(1,num_epochs+1):
        logging.info("\nEpoch n."+str(epoch) +":" )
        train_generator = SLC.read_split('training')
        next_token_tpl = None
        valid_generator = SLC.read_split('validation')
        epoch_valid_loss = inf
        try:
            while(True):
                current_token_tpl, next_token_tpl = get_tokens_tpls(next_token_tpl, train_generator, senseindices_db_c,
                                                                    vocab_h5, model.last_idx_senses)
                optimizer.zero_grad()
                loss = compute_loss_iteration(data, model, current_token_tpl, next_token_tpl)
                loss.backward()
                optimizer.step()
                losses.append(loss)

        except StopIteration:
            continue # next epoch

    getLossGraph(losses)

def getLossGraph(source1):
    plt.plot(source1, color='red', marker='o')
