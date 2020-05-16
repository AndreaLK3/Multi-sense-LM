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
    #Utils.init_logging('create_adjacencies_matrix_numpy.log')

    logging.info(graph_dataobj)
    tot_nodes = graph_dataobj.x.shape[0]

    edges_added_per_area = int(area_size ** 1.5)
    m = edges_added_per_area
    k = area_size
    tot_dim_row = area_size + 3 * m
    nodes_arraytable = np.ones(shape=(tot_nodes, tot_dim_row)) * -1

    for i in range(tot_nodes):
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



