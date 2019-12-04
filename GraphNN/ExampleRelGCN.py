import torch
import torch_geometric
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from WordEmbeddings.ComputeEmbeddings import Method
import os
import random
import numpy as np
import Utils
import logging
from torch_geometric.utils import convert
import networkx

# In this example, we do not extract the node features, word and sense vocabulary indices, etc.
# We use Random Number Generation to create a small structure.

# Utility function
def convert_dict_into_lts(dict_target_sources):
    t_s_lts = []
    for key in dict_target_sources.keys():
        for value in dict_target_sources[key]:
            t_s_lts.append((value, key))
    return t_s_lts

##### Edges in the simulation graph

# 1 definition (or example) refers to only 1 sense. 1 sense can have multiple definitions or examples
def setrandomedges_d_or_e(senses_low, senses_high, elements_low, elements_high):
    num_elements = elements_high - elements_low
    possible_elements_ls = list(range(elements_low, elements_high))
    sense_deflinks_dict = {}
    for sense_index in range(senses_low, senses_high):
        num_defs_for_sense = random.randint(1, 2) # or 1+num_elements//5 as the upper bound
        sense_defs_ls = np.random.choice(possible_elements_ls, num_defs_for_sense, replace=False)
        possible_elements_ls = list( set(possible_elements_ls).difference(set(sense_defs_ls)))
        sense_deflinks_dict[sense_index] = sense_defs_ls
    return convert_dict_into_lts(sense_deflinks_dict)

# 1 sense must be referred by exactly 1 global. 1 global can have multiple senses
def setrandomedges_sc(senses_low, senses_high, global_low, global_high):
    num_globals = global_high - global_low
    possible_senses_ls = list(range(senses_low, senses_high))
    possible_globals_ls = list(range(global_low, global_high))
    sense_globallinks_lts = []
    for sense in possible_senses_ls:
        sp_global = np.random.choice(possible_globals_ls, None, replace=False)
        sense_globallinks_lts.append((sp_global, sense))
    return sense_globallinks_lts


# Any global can have multiple synonyms (or antonyms). However, they must be symmetrical (a,b) ==> (b,a).
# And an antonym can not be a synonym
def setrandomedges_syn_or_ant(global_low, global_high, restrict=False, opposite_connections_lts=[]):
    num_globals = global_high - global_low
    possible_nyms_ls = list(range(global_low, global_high))
    globals_nyms_symmetric_lts = []
    for global_index in range(global_low, global_high):
        num_nyms_for_global = random.randint(0, min(2, num_globals))
        global_nyms_ls = np.random.choice(possible_nyms_ls, num_nyms_for_global, replace=False)
        for nym_index in global_nyms_ls:
            if nym_index != global_index: # no self-loops here
                globals_nyms_symmetric_lts.append((global_index, nym_index))
                globals_nyms_symmetric_lts.append((nym_index, global_index))
    if restrict==True:
        results_lts = list( set(globals_nyms_symmetric_lts).difference(set(opposite_connections_lts)))
    else:
        results_lts = globals_nyms_symmetric_lts
    return results_lts

######

##### The RGCN
# Modified from: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/rgcn.py
class NetRGCN(torch.nn.Module):
    def __init__(self, data):
        super(NetRGCN, self).__init__()
        # self.conv1 = RGCNConv(
        #     data.num_nodes, 16, dataset.num_relations, num_bases=30)
        # self.conv2 = RGCNConv(
        #     16, dataset.num_classes, dataset.num_relations, num_bases=30)
        self.conv1 = RGCNConv(
            in_channels=data.num_nodes, out_channels=16, num_relations=data.num_relations,
            num_bases=data.num_relations)

    def forward(self, edge_index, edge_type, edge_norm):
        # x = F.relu(self.conv1(None, edge_index, edge_type))
        # x = self.conv2(x, edge_index, edge_type)
        # return F.log_softmax(x, dim=1)



def createInputGraph():
    Utils.init_logging('temp.log')

    num_senses = 4
    num_sp = 2
    num_def = 8
    num_exs = 8
    NUM_NODES = num_senses + num_sp + num_def + num_exs
    NUM_NODE_FEATURES = 10
    NUM_RELATIONS = 5
    # X (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
    #
    # The nodes are: all the sense embeddings + all the single-prototype embeddings
    #               + all the sentence embeddings from the definitions and examples.
    # All the nodes must have an index, from 0 to num_nodes -1, and the same dimensionality, num_node_features.
    X_senses = torch.rand((num_senses, NUM_NODE_FEATURES))
    X_sp = torch.rand((num_sp, NUM_NODE_FEATURES))
    X_defs = torch.rand((num_def, NUM_NODE_FEATURES))
    X_exs = torch.rand((num_exs, NUM_NODE_FEATURES))

    # The order for the index of the nodes?
    # sense = [0,se) ; single prototype = [se,se+sp) ; definitions = [se+sp, se+sp+d) ; examples = [se+sp+d, e==num_nodes)
    # The index will be the row number in the matrix X

    # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
    # We can operate with a list of S-D tuples, adding t().contiguous()
    # The Procedure to set up both X and edge_index will be:
    # - read the archive of processed definitions. We encounter the sense_wn_id (e.g. active.n.03), and the def. text
    # - consider the database indices_table.sql. It has the columns: word_sense, vocab_index (from 0 to se),
    #   start_defs, end_defs, start_examples, end_examples.
    # - load (for instance) vectorized_FastText_definitions.npy. Even with > 20K vectors, it is very light at ~58MB
    # - (Using WordNet as a source, we always have 1 definition). Use [start_defs, end_defs) to extract the vector(s) for
    #   the definition of a sense, and append that vector to X_defs
    # - Register the connections: add a tuple (source=definition_index=sp+def, target=sense)
    # - Do the same for examples: get the examples' vectors for that sense from [start_examples, end_examples) in
    #   indices_table.sql, and get the corresponding rows of vectorized_FastText_examples.npy.
    # - Append vectors to submatrix X_exs (to be put together later), and add the connections [d, e==num_nodes) ->[0,se)

    logging.info("Senses : [ " + str(0) + " , " + str(num_senses) + ")")
    logging.info("Global single-prototypes : [ " + str(num_senses) + " , " + str(num_senses + num_sp) + ")")
    logging.info("Definitions : [ " + str(num_senses + num_sp) + " , " + str(num_senses + num_sp + num_def) + ")")
    logging.info("Examples : [ " + str(num_senses + num_sp + num_def) + " , " + str(NUM_NODES) + ")")


    # For this example, however, we initialized randomly the features of the vectors, and we proceed by
    # randomly determining connections between different kinds of nodes, in the way the task allows.
    # definitions -> senses : [se+sp, se+sp+d) -> [0,se)
    edges_defs = setrandomedges_d_or_e(0, num_senses, num_senses+num_sp, num_senses+num_sp+num_def)
    logging.info("Edges - def . Connections:" + str(edges_defs))

    # examples --> senses : [se+sp+d, e==num_nodes) -> [0,se)
    edges_exs = setrandomedges_d_or_e(0, num_senses, num_senses+num_sp+num_def, NUM_NODES)
    logging.info("Edges - exs . Connections:" + str(edges_exs))

    # global (a.k.a. single-prototype) -> senses : [se,se+sp) -> [0,se)
    edges_sc = setrandomedges_sc(0, num_senses, num_senses, num_senses+num_sp)
    logging.info("Edges - sc . Connections:" + str(edges_sc))

    # global -> global : [se,se+sp) -> [se,se+sp). Bidirectional (which means 2 connections, (a,b) and (b,a)
    edges_syn = setrandomedges_syn_or_ant(num_senses, num_senses+num_sp)
    logging.info("Edges - syn . Connections:" + str(edges_syn))
    edges_ant = setrandomedges_syn_or_ant(num_senses, num_senses+num_sp,
                                                     restrict=True, opposite_connections_lts=edges_syn)
    logging.info("Edges - ant . Connections:" + str(edges_ant))


    ##### Reuniting the components of the graph
    X = torch.cat([X_senses, X_sp, X_defs, X_exs])
    all_edges_lts = torch.tensor(edges_defs + edges_exs + edges_sc + edges_syn + edges_ant)
    edge_types = torch.tensor([0] * len(edges_defs) + [1] * len(edges_exs) + [2] * len(edges_sc) +
                              [3] * len(edges_syn) + [4] * len(edges_ant))

    all_edges = all_edges_lts.t().contiguous()
    graph = torch_geometric.data.Data(x=X, edge_index=all_edges, edge_type=edge_types, num_relations=NUM_RELATIONS)

    ##### Printing
    netx_graph = convert.to_networkx(graph)
    networkx.write_gexf(netx_graph, path=os.path.join('GraphNN', 'exampleGraph.gexf'))



