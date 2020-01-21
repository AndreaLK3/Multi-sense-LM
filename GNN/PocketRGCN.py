import torch
import torch_geometric
from torch_geometric.nn import RGCNConv
import torch.nn.functional as tF

import Graph.GraphArea as GA
import random
import Utils
import logging
import matplotlib.pyplot as plt
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# In this example, we do not extract the node features, word and sense vocabulary indices, etc.
# We use Random Number Generation to create a small structure.

def createInputGraph():
    Utils.init_logging('createInputGraph.log')

    num_senses = 10
    num_sp = 5
    num_def = 20
    num_exs = 20
    NUM_NODES = num_senses + num_sp + num_def + num_exs
    NUM_NODE_FEATURES = 100
    NUM_RELATIONS = 5
    # X (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features].
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

    # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges].
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
    logging.debug("Edges - def . Connections:" + str(edges_defs))

    # examples --> senses : [se+sp+d, e==num_nodes) -> [0,se)
    edges_exs = setrandomedges_d_or_e(0, num_senses, num_senses+num_sp+num_def, NUM_NODES)
    logging.debug("Edges - exs . Connections:" + str(edges_exs))

    # global (a.k.a. single-prototype) -> senses : [se,se+sp) -> [0,se)
    edges_sc = setrandomedges_sc(0, num_senses, num_senses, num_senses+num_sp)
    logging.debug("Edges - sc . Connections:" + str(edges_sc))

    # global -> global : [se,se+sp) -> [se,se+sp). Bidirectional (which means 2 connections, (a,b) and (b,a)
    edges_syn = setrandomedges_syn_or_ant(num_senses, num_senses+num_sp)
    logging.debug("Edges - syn . Connections:" + str(edges_syn))
    edges_ant = setrandomedges_syn_or_ant(num_senses, num_senses+num_sp,
                                                     restrict=True, opposite_connections_lts=edges_syn)
    logging.debug("Edges - ant . Connections:" + str(edges_ant))
    logging.info("Total number of edges = " + str(len(edges_defs)+len(edges_exs)+
                                                  len(edges_sc)+len(edges_syn)+len(edges_ant)) )


    ##### Reuniting the components of the graph
    X = torch.cat([X_senses, X_sp, X_defs, X_exs])
    all_edges_lts = torch.tensor(edges_defs + edges_exs + edges_sc + edges_syn + edges_ant)
    edge_types = torch.tensor([0] * len(edges_defs) + [1] * len(edges_exs) + [2] * len(edges_sc) +
                              [3] * len(edges_syn) + [4] * len(edges_ant))
    node_types = torch.tensor([0]*num_senses + [1]*num_sp + [2]*num_def + [3]*num_exs)

    all_edges = all_edges_lts.t().contiguous()
    graph = torch_geometric.data.Data(x=X,
                                      edge_index=all_edges,
                                      edge_type=edge_types,
                                      node_types=node_types,
                                      num_relations=NUM_RELATIONS)

    ##### Printing
    #netx_graph = convert.to_networkx(graph)
    #networkx.write_gexf(netx_graph, path=os.path.join('GNN', 'exampleGraph.gexf'))
    example_node_index = 12
    #get_neighbours(graph.edge_index, graph.edge_type, example_node_index)

    return graph


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

##### The GNN
class NetRGCN(torch.nn.Module):
    def __init__(self, data):
        super(NetRGCN, self).__init__()
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.conv1 = RGCNConv(in_channels=data.x.shape[1], # doc: "Size of each input sample "
                              # although, if I followed the example, it should be num_nodes
                              out_channels=data.x.shape[1], # doc: "Size of each output sample "
                              num_relations=data.num_relations,
                              num_bases=data.num_relations) #[N,h] ==[55,100]
        self.linear2global = torch.nn.Linear(in_features=data.x.shape[1], out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        self.linear2sense = torch.nn.Linear(in_features=data.x.shape[1], out_features=self.last_idx_senses, bias=True)

    def forward(self, batch_x, batch_edge_index, batch_edge_type):
        x_Lplus1 = tF.relu(self.conv1(batch_x, batch_edge_index, batch_edge_type))
        x1_current_node = x_Lplus1[0] # current_node_index
        logits_global = self.linear2global(x1_current_node) # shape=torch.Size([5])
        logits_sense = self.linear2sense(x1_current_node)

        return (tF.log_softmax(logits_global, dim=0), tF.log_softmax(logits_sense, dim=0))

### Training

# Given the tuple of numerical indices for an element, e.g. (13,5), retrieve a list
# of either 1 or 2 tuples that are input to the forward function, i.e. (x, edge_index, edge_type)
# Here i decide what is the starting token for a prediction. For now, it is "sense if present, else global"
def get_forwardinput_forelement(global_idx, sense_idx, node_area_size, graph):
    forward_input_ls = []
    if (sense_idx == -1):
        x, edge_index, edge_type = GA.get_graph_area(global_idx.item(), node_area_size, graph)
        forward_input_ls.append((x, edge_index, edge_type))
    else:
        x, edge_index, edge_type = GA.get_graph_area(sense_idx.item(), node_area_size, graph)
        forward_input_ls.append((x, edge_index, edge_type))
    return forward_input_ls

#n: will be modified in order to use the generator when in the full model instead of the pocket model
# returns a lists of indices that covers the current_tokens and the labels
def get_batch_indices_tpls(batch_size, training_dataset, current_train_index):
    batch_indices_lts = []
    for i in range(batch_size+1): # I also get here the label that comes after the first element
        (token_global_idx, token_sense_idx) = training_dataset[current_train_index+i]
        batch_indices_lts.append((token_global_idx, token_sense_idx))
    return batch_indices_lts


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

        (y_global_raw_idx, y_sense_idx) = input_indices_lts[i + 1]
        (y_global, y_sense) = (
            torch.Tensor([y_global_raw_idx - model.last_idx_senses]).type(torch.int64).to(DEVICE),
            torch.Tensor([y_sense_idx]).type(torch.int64).to(DEVICE))

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


def train():

    inputgraph_dataobject = createInputGraph()
    RGCN_modelobject = NetRGCN(inputgraph_dataobject)

    graph, model = inputgraph_dataobject.to(DEVICE), RGCN_modelobject.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    #out = model(data.edge_index, data.edge_type, data.edge_norm)
    training_dataset = torch.Tensor([(10,-1),(11,5),(12,-1),(13,3),(14,-1),(10,9),(11,-1),(12,-1),(13,-1),(14,-1),
                        (10,-1),(11,5),(12,-1),(13,3),(14,-1),(10,9),(11,5),(12,-1),(13,3),(14,-1),
                        (12,-1),(13,3),(10,9),(13,-1),(14,-1),(12,-1),(11,-1),(12,-1),(10,9),(14,-1)]).type(torch.int64)
    training_dataset.to(DEVICE, dtype=torch.int64)
    num_epochs = 20
    model.train()
    losses = []
    loss = 0
    batch_size = 4
    node_area_size = 8

    for epoch in range(1,num_epochs+1):
        logging.info("\nEpoch n."+str(epoch) +":" )

        for batch_start in range(0, len(training_dataset) - batch_size, batch_size - 1):
            optimizer.zero_grad()
            logging.info('\n-----\nLocation=' + str(batch_start))

            input_indices_lts = get_batch_indices_tpls(batch_size, training_dataset, batch_start)
            logging.debug(input_indices_lts)
            batch_rgcn_input_ls = []
            for i in range(len(input_indices_lts)-1):
                (global_idx, sense_idx) = input_indices_lts[i]
                batch_rgcn_input_ls.extend(get_forwardinput_forelement(global_idx, sense_idx, node_area_size, graph))
            loss = compute_batch_losses(input_indices_lts, batch_rgcn_input_ls, model)
            loss.backward()
            optimizer.step()

        logging.info(loss)
        losses.append(loss)

    getLossGraph(losses)

def getLossGraph(source1):
    plt.plot(source1, color='red', marker='o')





