import logging

import numpy as np
import torch
import Utils
from time import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Find the node's neighbours and more, to select the graph_area
# Increase gradually the hop distance, from i=1
# •	At Hop distance d=i , retrieve, in order: definitions, examples, synonyms, antonyms
# Stop when the maximum number of nodes is reached (as defined by the input dimensions of the RGCN)
# n: Retrieving N nodes also includes the starting node
def get_indices_area_toinclude(edge_index, edge_type, node_index, area_size, max_hops):

    nodes_retrieved_lls = [[node_index], []]
    current_hops = 1
    num_to_retrieve = area_size
    stop_flag = False

    edges_retrieved_set = set()

    while current_hops <= max_hops and stop_flag==False:
        start_idx = 0
        nodes_queue_at_current_level = nodes_retrieved_lls[current_hops - 1]
        nodes_queue_at_next_hop = nodes_retrieved_lls[current_hops]
        while(start_idx < len(nodes_queue_at_current_level)):
            start_node = nodes_queue_at_current_level[start_idx]
            start_idx = start_idx + 1
            node_edges, indices_of_edges_with_node = get_node_edges(edge_index, edge_type, start_node)

            edges_retrieved_set = edges_retrieved_set.union(set(indices_of_edges_with_node))
            new_nodes = list(map(lambda edge_tpl: edge_tpl[1], node_edges)) + list(
                map(lambda edge_tpl: edge_tpl[0], node_edges))
            for n in new_nodes:
                if sum([len(hop_ls) for hop_ls in nodes_retrieved_lls]) >= num_to_retrieve:
                    stop_flag = True
                    break
                if n not in nodes_queue_at_next_hop and not n in [n for hop_nodes_ls in nodes_retrieved_lls for n in hop_nodes_ls]:
                    nodes_queue_at_next_hop.append(n)

        nodes_retrieved_lls.append([])
        current_hops = current_hops + 1

    logging.info("start_node=" + str(node_index) + " , nodes_retrieved= " + str(nodes_retrieved_lls))
    return [n for hop_nodes_ls in nodes_retrieved_lls for n in hop_nodes_ls], list(edges_retrieved_set)

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


### Entry point function to get input - either the whole graph area, or the immediate neighbours
def get_grapharea_elements(starting_node_index, area_size, graph, hops_in_area):
    logging.debug("starting_node_index=" + str(starting_node_index))

    node_indices_ls, all_edges_retrieved_ls = get_indices_area_toinclude(graph.edge_index, graph.edge_type, starting_node_index, area_size, hops_in_area)

    # original time: t5 - t4 = 1.54 s; version 3 time: 0.05 s
    edges_retrieved_ls = list(filter(lambda edge_idx: graph.edge_index[0][edge_idx].item() in set(node_indices_ls)
                                                      and graph.edge_index[1][edge_idx].item() in set(node_indices_ls),
                                     all_edges_retrieved_ls))  # to include an edge, both source and target node must be in the batch
    edges_indices = torch.tensor(sorted(edges_retrieved_ls)).to(torch.int64).to(DEVICE)
    edges_defaultValues = graph.edge_index.t().index_select(0, edges_indices)
    logging.debug("edges_defaultValues= " + str(edges_defaultValues))
    edges_reindexed = []
    for (src, trg) in edges_defaultValues:
        src_01 = node_indices_ls.index(src)
        trg_01 = node_indices_ls.index(trg)
        edges_reindexed.append((src_01, trg_01))
    logging.debug("edges_reindexed= " + str(edges_reindexed))

    with torch.no_grad():
        area_edge_index = torch.autograd.Variable(torch.tensor(edges_reindexed).t().to(torch.int64)).to(DEVICE)

    area_edge_type = graph.edge_type.index_select(0, index=edges_indices).to(torch.int64).to(DEVICE)

    return (node_indices_ls, area_edge_index, area_edge_type)

