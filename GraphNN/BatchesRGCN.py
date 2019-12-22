import logging

import numpy as np
import torch
import Utils


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_indices_toinclude(edge_index, edge_type, node_index, num_to_retrieve):
    nodes_retrieved = [node_index]
    start_idx = 0
    stop_flag = False
    while(start_idx < len(nodes_retrieved) and stop_flag==False):
        start_node = nodes_retrieved[start_idx]
        start_idx = start_idx + 1

        new_edges = get_neighbours(edge_index, edge_type, start_node)
        new_nodes = list(map(lambda edge_tpl: edge_tpl[1], new_edges)) + list(map(lambda edge_tpl: edge_tpl[0], new_edges))
        for n in new_nodes:
            if len(nodes_retrieved) >= num_to_retrieve:
                stop_flag = True
                break
            if n not in nodes_retrieved:
                nodes_retrieved.append(n)
                logging.debug("start_node=" + str(start_node) + " , nodes_retrieved= " + str(nodes_retrieved))

    return nodes_retrieved


def get_neighbours(edge_index, edge_type, node_index):
    indices_of_edges_where_node_is_source = np.where(edge_index[0].cpu().numpy() == node_index)[0]
    indices_of_edges_where_node_is_target = np.where(edge_index[1].cpu().numpy() == node_index)[0]
    node_neighbours_edges = []
    for i in np.concatenate([indices_of_edges_where_node_is_source,indices_of_edges_where_node_is_target]):
        node_neighbours_edges.append((edge_index[0][i].item(),edge_index[1][i].item(),edge_type[i].item()))
    node_neighbours_edges = sorted(list(set(node_neighbours_edges)), key=lambda src_trg_type_tpl: src_trg_type_tpl[2])
    logging.debug("node_index=" + str(node_index) + " -> node_neighbours_edges=" + str(node_neighbours_edges))
    return node_neighbours_edges


def get_batch_of_graph(starting_node_index, batch_size, graph):
    logging.debug("starting_node_index=" + str(starting_node_index))
    batch_elements_indices_ls = sorted(get_indices_toinclude(graph.edge_index, graph.edge_type, starting_node_index, batch_size))
    batch_elements_indices = torch.Tensor(batch_elements_indices_ls).to(torch.int64).to(DEVICE)
    batch_x = graph.x.index_select(0, batch_elements_indices)

    edges_indices = torch.Tensor([e_col for e_col in range(len(graph.edge_index[0])) # it can be tensor([]) in some cases
                    if (graph.edge_index[0][e_col] in batch_elements_indices
                        and graph.edge_index[1][e_col] in batch_elements_indices)]).to(torch.int64).to(DEVICE)
    #for edge_index in edges_indices:
    #    print(graph.edge_index.t()[edge_index])
    batch_edge_index_lts_defaultValues = graph.edge_index.t().index_select(0, edges_indices)

    elem_edges_lts_reindexed = []
    for (src, trg) in batch_edge_index_lts_defaultValues:
        src_01 = batch_elements_indices_ls.index(src)
        trg_01 = batch_elements_indices_ls.index(trg)
        elem_edges_lts_reindexed.append((src_01, trg_01))

    with torch.no_grad():
        batch_edge_index = torch.autograd.Variable(torch.Tensor(elem_edges_lts_reindexed).t().to(torch.int64)).to(DEVICE)

    batch_edge_type = graph.edge_type.index_select(0, index=edges_indices).to(torch.int64).to(DEVICE)

    return (batch_x, batch_edge_index, batch_edge_type)