import torch
import Graph.GraphArea as GA
import Graph.DefineGraph as DG
import Filesystem as F
import numpy as np
import logging
import Utils
import os
from scipy import sparse
import pandas as pd
from GNN.Models.Common import lemmatize_node
from types import SimpleNamespace


# Utility function: determine which globals have more than 1 sense, versus the dummySenses and 0or1 sense.
# (We evaluate the number of senses of the lemmatized form). Used to compute different Perpexities
def compute_globals_numsenses(graph_dataobj, grapharea_matrix, grapharea_size):
    num_senses_ls = []
    last_idx_senses = graph_dataobj.node_types.tolist().index(1)
    last_idx_globals = graph_dataobj.node_types.tolist().index(2)
    first_idx_dummySenses = Utils.get_startpoint_dummySenses()
    logging.info("Examining the graph + corpus, to determine which globals have multiple senses...")

    vocab_fpath = os.path.join("Vocabulary", "vocabulary_of_globals.h5");
    vocabulary_df = pd.read_hdf(vocab_fpath)
    vocabulary_wordList = vocabulary_df['word'].to_list().copy()
    vocabulary_lemmatizedWordsList = vocabulary_df['lemmatized_form'].to_list().copy()
    vocabulary_frequencyList = vocabulary_df['frequency'].to_list().copy()

    log_idx = 10000
    # iterate over the globals
    for idx in range(last_idx_senses, last_idx_globals):
        if idx % log_idx == 0: logging.info("global idx=" + str(idx) + "...")
        adjacent_nodes, edges, edge_type = get_node_data(grapharea_matrix, idx, grapharea_size, features_mask=(True, True, True))

        args_dict = {'first_idx_dummySenses': first_idx_dummySenses, 'last_idx_senses':last_idx_senses,
                     'vocabulary_wordlist':vocabulary_wordList, 'vocabulary_lemmatizedList':vocabulary_lemmatizedWordsList,
                     'grapharea_matrix':grapharea_matrix, 'grapharea_size':grapharea_size}  # packing the parameters for the lemmatizer function
        args = SimpleNamespace(**args_dict)
        adjacent_nodes, edges, edge_type = lemmatize_node(adjacent_nodes, edges, edge_type, args)

        num_senses = edge_type.tolist().count(2)
        num_dummy_senses = len(list(filter(lambda n: first_idx_dummySenses < n and n < last_idx_senses, adjacent_nodes)))
        num_senses = num_senses - num_dummy_senses

        logging.debug("word=" + str(vocabulary_wordList[idx-last_idx_senses]) +
                         " ; edge_type="+str(edge_type) + " ; num_senses=" + str(num_senses))
        num_senses_ls.append(num_senses)

    new_vocabulary_data = list(zip(vocabulary_wordList, vocabulary_frequencyList, vocabulary_lemmatizedWordsList, num_senses_ls))
    new_vocabulary_df = pd.DataFrame(data=new_vocabulary_data, columns=['word','frequency','lemmatized_form','num_senses'])
    vocabulary_h5_archive = pd.HDFStore(vocab_fpath, mode='w')
    min_itemsize_dict = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'frequency': Utils.HDF5_BASE_SIZE_512 / 8,
                         'lemmatized_form': Utils.HDF5_BASE_SIZE_512 / 4, 'num_senses': Utils.HDF5_BASE_SIZE_512 / 8}
    vocabulary_h5_archive.append(key='vocabulary', value=new_vocabulary_df, min_itemsize=min_itemsize_dict)
    vocabulary_h5_archive.close()


def get_multisense_globals_indices():
    vocab_fpath = os.path.join("Vocabulary", "vocabulary_of_globals.h5");
    vocabulary_df = pd.read_hdf(vocab_fpath)
    vocabulary_num_senses_ls = vocabulary_df['num_senses'].to_list().copy()

    multisense_globals_indices = [i for i in range(len(vocabulary_num_senses_ls)) if vocabulary_num_senses_ls[i] > 1]

    return multisense_globals_indices



### Getter function, to extract node area data from a row in the matrix
def get_node_data(grapharea_matrix, i, grapharea_size, features_mask=(True,True,True)):
    CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())
    k = grapharea_size
    m = _edges_added_per_area = int(grapharea_size ** 1.5)
    nodes=None; edgeindex=None; edgetype=None
    if features_mask[0]==True:
        # Accessing sparse matrix. Everything was shifted +1, so now: we ignore 0 ; we shift -1; we get the data
        nodes_ls =list(map(lambda value: value - 1, filter(lambda num: num != 0, grapharea_matrix[i, 0:k].todense().tolist()[0])))
        nodes = torch.tensor(nodes_ls).to(torch.long).to(CURRENT_DEVICE)

    if features_mask[1] == True:
        edgeindex_sources_ls = list(map( lambda value: value-1, filter(lambda num: num != 0,
                                                                       grapharea_matrix[i, k:k + m].todense().tolist()[0])))
        edgeindex_targets_ls = list(map( lambda value: value-1, filter(lambda num: num != 0,
                                                                       grapharea_matrix[i, k + m:k + 2 * m].todense().tolist()[0])))
        edgeindex = torch.tensor([edgeindex_sources_ls, edgeindex_targets_ls]).to(torch.int64).to(CURRENT_DEVICE)

    if features_mask[2] == True:
        edgetype_ls = list(map( lambda value: value-1, filter(lambda num: num != 0,
                                                              grapharea_matrix[i, k + 2 * m: k + 3 * m].todense().tolist()[0])))
        edgetype = torch.tensor(edgetype_ls).to(torch.int64).to(CURRENT_DEVICE)

    return nodes, edgeindex, edgetype


### Creation function - numpy version
def create_adjacencies_matrix_numpy(graph_dataobj, area_size, hops_in_area):
    Utils.init_logging('create_adjacencies_matrix_numpy.log')

    logging.info(graph_dataobj)
    tot_nodes = graph_dataobj.x.shape[0]

    edges_added_per_area = int(area_size ** 1.5)
    m = edges_added_per_area
    k = area_size
    tot_dim_row = area_size + 3 * m
    nodes_arraytable = np.ones(shape=(tot_nodes, tot_dim_row)) * -1
    for i in range(tot_nodes):
        try:  # debug
            node_index = i
            (adj_nodes_ls, adj_edge_index, adj_edge_type) = GA.get_grapharea_elements(node_index, area_size, graph_dataobj, hops_in_area)
            if i % 1000 == 0:
                logging.info("node_index=" + str(node_index))
            # extract sources and targets from the edge_index related to the node
            adj_edge_sources = adj_edge_index[0]
            adj_edge_targets = adj_edge_index[1]

            # convert
            arr_adj_edge_sources = adj_edge_sources.cpu().numpy()
            arr_adj_edge_targets = adj_edge_targets.cpu().numpy()
            arr_adj_edge_type = adj_edge_type.cpu().numpy()

            # assign at the appropriate locations
            nodes_arraytable[i][0:len(adj_nodes_ls)] = np.array(adj_nodes_ls)
            nodes_arraytable[i][k: k + min(len(arr_adj_edge_sources), m)] = arr_adj_edge_sources[0:m]
            nodes_arraytable[i][k + m: k + m + min(len(arr_adj_edge_targets), m)] = arr_adj_edge_targets[0:m]
            nodes_arraytable[i][k + 2 * m: k + 2 * m + min(len(arr_adj_edge_type), m)] = arr_adj_edge_type[0:m]
        except Exception:
            logging.info("graph_dataobj="+str(graph_dataobj))
            logging.info("node_index=" + str(node_index))
            logging.info("adj_nodes_ls="+str(adj_nodes_ls))
            logging.info("adj_edge_index=" + str(adj_edge_index))
            logging.info("adj_edge_type=" + str(adj_edge_type))

    return nodes_arraytable

### Entry point function. Temporarily modified. Numpy version.
def get_grapharea_matrix(graphdata_obj, area_size, hops_in_area):

    candidate_fnames = [fname for fname in os.listdir(F.FOLDER_GRAPH)
                        if ((fname.endswith(F.GRAPHAREA_FILE)) and ('nodes_' + str(area_size) + '_areahops_' + str(hops_in_area) + '_' in fname))]
    if len(candidate_fnames) == 0:
        logging.info("Pre-computing and saving graphArea matrix, with area_size=" + str(area_size))
        grapharea_matrix = create_adjacencies_matrix_numpy(graphdata_obj, area_size, hops_in_area)
        out_fpath = os.path.join(F.FOLDER_GRAPH,
                                 'nodes_' + str(area_size) + '_areahops_' + str(hops_in_area) + '_' + F.GRAPHAREA_FILE)
        grapharea_matrix = grapharea_matrix + 1 # shift the matrix of +1, storage default element will be 0 and not -1
        coo_mat = sparse.coo_matrix(grapharea_matrix)
        csr_mat = coo_mat.tocsr()
        sparse.save_npz(out_fpath, csr_mat)
    else:
        fpath = os.path.join(F.FOLDER_GRAPH, candidate_fnames[0]) # we expect to find only one
        logging.info("Loading graphArea matrix, with area_size=" + str(area_size) + " from: " + str(fpath))
        csr_mat = sparse.load_npz(fpath)

    return csr_mat



