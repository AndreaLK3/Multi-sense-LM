import logging

import numpy as np
import torch
import Utils
from time import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Find the node's neighbours, to use for batching:
# Increase gradually the hop distance, from i=1
# •	At Hop distance d=i , retrieve, in order: definitions, examples, synonyms, antonyms
# Stop when the maximum number of nodes is reached (as defined by the input dimensions of the RGCN)
# n: Retrieving N nodes also includes the starting node
def get_indices_toinclude(edge_index, edge_type, node_index, num_to_retrieve):
    nodes_retrieved = [node_index]
    start_idx = 0
    stop_flag = False
    edges_retrieved_set = set()
    while(start_idx < len(nodes_retrieved) and stop_flag==False):
        start_node = nodes_retrieved[start_idx]
        start_idx = start_idx + 1

        node_edges, indices_of_edges_with_node = get_node_edges(edge_index, edge_type, start_node)
        # logging.info('node_index='+ str(node_index) + '; indices_of_edges_with_node' + str(indices_of_edges_with_node))
        edges_retrieved_set = edges_retrieved_set.union(set(indices_of_edges_with_node))
        new_nodes = list(map(lambda edge_tpl: edge_tpl[1], node_edges)) + list(map(lambda edge_tpl: edge_tpl[0], node_edges))
        for n in new_nodes:
            if len(nodes_retrieved) >= num_to_retrieve:
                stop_flag = True
                break
            if n not in nodes_retrieved:
                nodes_retrieved.append(n)
                logging.debug("start_node=" + str(start_node) + " , nodes_retrieved= " + str(nodes_retrieved))

    return nodes_retrieved, list(edges_retrieved_set)

# Auxiliary function: find the immediate neighbours of a node, in the given order. Both directions of edges are included
# edge_type = def:0, exs:1, sc:2, syn:3, ant:4
def get_node_edges(edge_index, edge_type, node_index):
    indices_of_edges_where_node_is_source = np.where(edge_index[0].cpu().numpy() == node_index)[0]
    indices_of_edges_where_node_is_target = np.where(edge_index[1].cpu().numpy() == node_index)[0]
    indices_of_edges_with_node = np.concatenate([indices_of_edges_where_node_is_source,indices_of_edges_where_node_is_target])
    node_edges = []
    for i in indices_of_edges_with_node:
        node_edges.append((edge_index[0][i].item(),edge_index[1][i].item(),edge_type[i].item()))
    node_edges = sorted(list(set(node_edges)), key=lambda src_trg_type_tpl: src_trg_type_tpl[2])
    logging.debug("node_index=" + str(node_index) + " -> node_neighbours_edges=" + str(node_edges))
    return node_edges, indices_of_edges_with_node


### Entry point function
def get_graph_area(starting_node_index, area_size, graph):
    logging.debug("starting_node_index=" + str(starting_node_index))
    node_indices_ls, all_edges_retrieved_ls = get_indices_toinclude(graph.edge_index, graph.edge_type, starting_node_index, area_size)
    node_indices = torch.Tensor(sorted(node_indices_ls)).to(torch.int64).to(DEVICE)
    area_x = graph.x.index_select(0, node_indices)

    # original time: t5 - t4 = 1.54 s; version 3 time: 0.05 s
    edges_retrieved_ls = list(filter(lambda edge_idx: graph.edge_index[0][edge_idx].item() in set(node_indices_ls)
                                                  and graph.edge_index[1][edge_idx].item() in set(node_indices_ls),
                                     all_edges_retrieved_ls)) # to include an edge, both source and target node must be in the batch
    edges_indices = torch.Tensor(sorted(edges_retrieved_ls)).to(torch.int64).to(DEVICE)
    edge_index_lts_defaultValues = graph.edge_index.t().index_select(0, edges_indices)

    elem_edges_lts_reindexed = []
    for (src, trg) in edge_index_lts_defaultValues:
        src_01 = node_indices_ls.index(src)
        trg_01 = node_indices_ls.index(trg)
        elem_edges_lts_reindexed.append((src_01, trg_01))

    with torch.no_grad():
        area_edge_index = torch.autograd.Variable(torch.Tensor(elem_edges_lts_reindexed).t().to(torch.int64)).to(DEVICE)

    area_edge_type = graph.edge_type.index_select(0, index=edges_indices).to(torch.int64).to(DEVICE)
    return (area_x, area_edge_index, area_edge_type)


### Utility function: given a LLS of reachable nodes from each starting point in the batch,
### include as many as it is necessary to:
# 1) guarantee propagation in the RGCN
# 2) return exactly the number of rows needed by the input matrix to the RGCN
def include_adjacent_nodes(adjacent_nodes_lls, size_X):

    logging.info('adjacent_nodes_lls=' + str(adjacent_nodes_lls))
    nodes_to_include = []
    level = 0
    end_flag = False
    while not(end_flag):
        for n in range(len(adjacent_nodes_lls)):
            node = adjacent_nodes_lls[n][level]
            nodes_to_include.append(node)
            if len(nodes_to_include) >= size_X:
                end_flag = True
                break
        level = level +1
    logging.info('size_X=' + str(size_X))
    logging.info('nodes_to_include='+str(nodes_to_include))
    return nodes_to_include


### Entry point function - batch mode
def get_batch_grapharea(batch_in_tokens_ls, size_X, graph):
    logging.info("starting_node_indices=" + str(batch_in_tokens_ls))
    neighbours_lls = []
    all_edges_retrieved_ls = []
    adjacency_ls_size = 2* (size_X // len(batch_in_tokens_ls))
    for starting_node_index in batch_in_tokens_ls:
        node_indices_ls, all_edges_ls = get_indices_toinclude(graph.edge_index, graph.edge_type,
                                                              starting_node_index, adjacency_ls_size)
        neighbours_lls.append(node_indices_ls)
        all_edges_retrieved_ls.extend(all_edges_ls) # must filter for useful edges later

    nodes_indices_ls = include_adjacent_nodes(neighbours_lls, size_X)
    nodes_indices = torch.Tensor(nodes_indices_ls).to(torch.int64).to(DEVICE)
    area_x = graph.x.index_select(0, nodes_indices)
    # there is a correspondence area_x[0] == graph.x[batch_in_tokens_ls[0]]
    logging.info('Correspondence=' + str(area_x[0] == graph.x[batch_in_tokens_ls[0]]))

    all_edges_retrieved_ls = sorted(list(set(all_edges_retrieved_ls)))

    edges_retrieved_indices = list(
        filter(lambda edge_idx: graph.edge_index[0][edge_idx].item() in set(nodes_indices_ls)
                                and graph.edge_index[1][edge_idx].item() in set(nodes_indices_ls),
               all_edges_retrieved_ls))  # to include an edge, both source and target node must be in the batch

    # The rest of the code is identical to the function defined previously:
    edges_indices = torch.Tensor(edges_retrieved_indices).to(torch.int64).to(DEVICE)

    edge_index_lts_defaultValues = graph.edge_index.t().index_select(0, edges_indices)

    elem_edges_lts_reindexed = []
    for (src, trg) in edge_index_lts_defaultValues:
        src_01 = nodes_indices_ls.index(src)
        trg_01 = nodes_indices_ls.index(trg)
        elem_edges_lts_reindexed.append((src_01, trg_01))

    with torch.no_grad():
        area_edge_index = torch.autograd.Variable(torch.Tensor(elem_edges_lts_reindexed).t().to(torch.int64)).to(DEVICE)

    area_edge_type = graph.edge_type.index_select(0, index=edges_indices).to(torch.int64).to(DEVICE)
    return (area_x, area_edge_index, area_edge_type)