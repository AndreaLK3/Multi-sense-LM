import torch
import Graph.GraphArea as GA
import Filesystem as F
import numpy as np
import os
from scipy import sparse
import logging

### Utility function, used in several places


def lemmatize_node(x_indices, edge_index, edge_type, model):
    currentglobal_relative_X_idx = x_indices[0]
    currentglobal_absolute_vocab_idx = currentglobal_relative_X_idx - model.last_idx_senses
    word = model.vocabulary_wordList[currentglobal_absolute_vocab_idx]
    lemmatized_word = model.vocabulary_lemmatizedList[currentglobal_absolute_vocab_idx]

    logging.debug("***\nword=" + str(word) + " ; lemmatized_word= "+ str(lemmatized_word))

    num_dummy_senses = len(list(filter(lambda n: model.first_idx_dummySenses < n and n < model.last_idx_senses, x_indices)))

    # if a word has edges that are not all self-loops, do not lemmatize it (to avoid turning 'as' into 'a')
    if len(edge_type)>num_dummy_senses:
        logging.debug("word has edges that are not all connections to dummySenses. We don't lemmatize")
        return x_indices, edge_index, edge_type
    if lemmatized_word != word:  # if the lemmatized word is actually different from the original, get the data
        try:
            logging.debug("Getting the data for the lemmatized word")
            lemmatized_word_absolute_idx = model.vocabulary_wordList.index(lemmatized_word)
            lemmatized_word_relative_idx = lemmatized_word_absolute_idx + model.last_idx_senses
            (x_indices_lemmatized, edge_index_lemmatized, edge_type_lemmatized) = \
                AD.get_node_data(model.grapharea_matrix, lemmatized_word_relative_idx, model.grapharea_size)
            return x_indices_lemmatized, edge_index_lemmatized, edge_type_lemmatized
        except ValueError:
            # the lemmatized word was not found in the vocabulary.
            logging.debug("The lemmatized word was not found in the vocabulary")
            return x_indices, edge_index, edge_type
    else:
        return x_indices, edge_index, edge_type


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

    logging.info(graph_dataobj)
    tot_nodes = graph_dataobj.x.shape[0]

    edges_added_per_area = int(area_size ** 1.5)
    m = edges_added_per_area
    k = area_size
    tot_dim_row = area_size + 3 * m
    nodes_arraytable = np.ones(shape=(tot_nodes, tot_dim_row)) * -1
    for i in range(tot_nodes):
        # try:  # debug
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

    return nodes_arraytable

### Entry point function. Temporarily modified. Numpy version.
def get_grapharea_matrix(graphdata_obj, area_size, hops_in_area, graph_folder, new=False):

    candidate_fnames = [fname for fname in os.listdir(graph_folder)
                        if ((fname.endswith(F.GRAPHAREA_FILE)) and ('nodes_' + str(area_size) + '_areahops_' + str(hops_in_area) + '_' in fname))]
    if len(candidate_fnames) == 0 or new:
        logging.info("Pre-computing and saving graphArea matrix, with area_size=" + str(area_size))
        grapharea_matrix = create_adjacencies_matrix_numpy(graphdata_obj, area_size, hops_in_area)
        out_fpath = os.path.join(graph_folder,
                                 'nodes_' + str(area_size) + '_areahops_' + str(hops_in_area) + '_' + F.GRAPHAREA_FILE)
        grapharea_matrix = grapharea_matrix + 1 # shift the matrix of +1, storage default element will be 0 and not -1
        coo_mat = sparse.coo_matrix(grapharea_matrix)
        csr_mat = coo_mat.tocsr()
        sparse.save_npz(out_fpath, csr_mat)
    else:
        fpath = os.path.join(graph_folder, candidate_fnames[0]) # we expect to find only one
        logging.info("Loading graphArea matrix, with area_size=" + str(area_size) + " from: " + str(fpath))
        csr_mat = sparse.load_npz(fpath)

    return csr_mat
