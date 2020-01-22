import torch
import Graph.GraphArea as GA
import Graph.DefineGraph as DG
import Filesystem as F
import numpy as np
import logging
import Utils
import os
from time import time


### Auxiliary getter function, to extract node area data from a row in the matrix
def get_node_data(grapharea_matrix, i, grapharea_size, edges_added_per_node=64):
    k = grapharea_size
    m = k * edges_added_per_node

    x_ls = list(filter( lambda num: num!=-1, grapharea_matrix[i][0:k]))
    edgeindex_sources_ls = list(filter( lambda num: num!=-1, grapharea_matrix[i][k:k + m] ))
    edgeindex_targets_ls = list(filter( lambda num: num!=-1, grapharea_matrix[i][k + m:k + 2*m ] ))
    edgetype_ls = list(filter( lambda num: num!=-1, grapharea_matrix[i][k + 2 * m: k + 3 * m ] ))

    x = torch.Tensor(x_ls)# .to(torch.int64)
    edgeindex = torch.Tensor([edgeindex_sources_ls, edgeindex_targets_ls])# .to(torch.int64)
    edgetype = torch.Tensor(edgetype_ls) # .to(torch.int64)

    return x, edgeindex, edgetype



### Creation function
def create_adjacencies_matrix(graph_dataobj, area_size, edges_added_per_node=64):

    out_fpath = os.path.join(F.FOLDER_GRAPH, 'nodes_' + str(area_size) + '_' + F.GRAPHAREA_FILE)
    # out_file = open(out_fpath, 'wb') -- used with numpy

    logging.info(graph_dataobj)
    tot_nodes = graph_dataobj.x.shape[0]

    k = area_size
    m = k * edges_added_per_node
    tot_dim_row = area_size + 3 * m
    nodes_arraytable = torch.ones(size=(tot_nodes, tot_dim_row), dtype=torch.int64) * -1

    # Given k = graph_area_size and m = max_edges, each row of the .npy array will have the following boundaries:
    # [0 : k) for the nodes.
    # [k: k+m) for the sources of edge_index
    # [k+m : k+2m) for the targets of edge_index
    # [k+2m : k+3m) for the edge_type.
    for i in range(tot_nodes):
        node_index = i
        (adj_nodes_ls, adj_edge_index, adj_edge_type) = GA.get_grapharea_elements(node_index, area_size, graph_dataobj)
        # padding the adjacent_nodes section to k, the graph_area_size
        arr_adj_nodes = torch.Tensor(adj_nodes_ls)
        # extract sources and targets from the edge_index related to the node
        adj_edge_sources = adj_edge_index[0]
        adj_edge_targets = adj_edge_index[1]

        # convert to numpy arrays
        arr_adj_edge_sources = adj_edge_sources.to(torch.int64)
        arr_adj_edge_targets = adj_edge_targets.to(torch.int64)
        arr_adj_edge_type = adj_edge_type.to(torch.int64)

        # assign at the appropriate locations
        nodes_arraytable[i][0:len(arr_adj_nodes)] = arr_adj_nodes
        nodes_arraytable[i][k:k+len(arr_adj_edge_sources)] = arr_adj_edge_sources
        nodes_arraytable[i][k+m:k+m+len(arr_adj_edge_targets)] = arr_adj_edge_targets
        nodes_arraytable[i][k+2*m: k+2*m+len(arr_adj_edge_type)] = arr_adj_edge_type

    torch.save(nodes_arraytable, out_fpath)
    # out_file.close() -- used with numpy
    return nodes_arraytable


### Entry point function
def get_grapharea_matrix(graphdata_obj, area_size):
    candidate_fnames = [fname for fname in os.listdir(F.FOLDER_GRAPH)
                        if ((F.GRAPHAREA_FILE in fname) and ('nodes_' + str(area_size) + '_' in fname))]
    if len(candidate_fnames) == 0:
        logging.info("Pre-computing and saving graphArea matrix, with area_size=" + str(area_size))
        grapharea_matrix = create_adjacencies_matrix(graphdata_obj, area_size)
    else:
        fpath = os.path.join(F.FOLDER_GRAPH, candidate_fnames[0]) # we expect to find only one
        logging.info("Loading graphArea matrix, with area_size=" + str(area_size) + " from: " + str(fpath))
        grapharea_matrix = torch.load(fpath) # -- used with numpy: allow_pickle=True
    return grapharea_matrix



