import torch
import torch.nn.functional as tfunc
import GNN.SenseLabeledCorpus as SLC
import GNN.NumericalIndices as NI
import logging
from Utils import DEVICE
import Graph.Adjacencies as AD


# Auxiliary function to pack an input tuple (x_indices, edge_index, edge_type)
# into a tensor [x_indices; edge_sources; edge_destinations; edge_type]
def pack_input_tuple_into_tensor(input_tuple, graph_area, max_edges=128):
    in_tensor = - 1 * torch.ones(size=(graph_area + max_edges*3,)).to(torch.long)
    x_indices = input_tuple[0]
    edge_sources = input_tuple[1][0]
    edge_destinations = input_tuple[1][1]
    edge_type = input_tuple[2]
    in_tensor[0:len(x_indices)] = x_indices
    in_tensor[graph_area: graph_area+len(edge_sources)] = edge_sources
    in_tensor[graph_area+max_edges:graph_area+max_edges+len(edge_destinations)] = edge_destinations
    in_tensor[graph_area+2*max_edges:graph_area+2*max_edges+len(edge_type)] = edge_type
    #logging.info("packing. in_tensor=" + str(in_tensor))
    return in_tensor


# When automatic batching is enabled, collate_fn is called with a list of data samples at each time.
# It is expected to collate the input samples into a batch for yielding from the data loader iterator.
class BPTTBatchCollator():

    def __init__(self, grapharea_size, sequence_length):
        self.grapharea_size = grapharea_size
        self.sequence_length = sequence_length

    def __call__(self, data): # This was collate_fn

        input_lls = []
        labels_ls = []

        i = 0
        input_ls = []
        for ((x, edge_index, edge_type), label_next_token_tpl) in data:
            if i >= self.sequence_length:
                input_lls.append(torch.stack(input_ls, dim=0))
                i=0
                input_ls = []

            input_ls.append(pack_input_tuple_into_tensor((x, edge_index, edge_type), self.grapharea_size))
            i = i + 1
            labels_ls.append(torch.tensor(label_next_token_tpl).to(torch.int64).to(DEVICE))
        # add the last one
        input_lls.append(torch.stack(input_ls, dim=0))
        return torch.stack(input_lls, dim=0), torch.stack(labels_ls, dim=0)



##### The Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, split_name, senseindices_db_c, vocab_h5, model,
                       grapharea_matrix, area_size, graph_dataobj):
        self.split_name = split_name
        self.generator = SLC.read_split(self.split_name)
        self.senseindices_db_c = senseindices_db_c
        self.vocab_h5 = vocab_h5
        self.gnn_model = model
        self.counter = 0

        self.grapharea_matrix = grapharea_matrix
        self.area_size = area_size
        self.graph_dataobj = graph_dataobj
        self.next_token_tpl = None


    def __getitem__(self, index):
        self.current_token_tpl, self.next_token_tpl = NI.get_tokens_tpls(self.next_token_tpl , self.generator,
                                                               self.senseindices_db_c, self.vocab_h5,
                                                               self.gnn_model.last_idx_senses)

        global_idx, sense_idx = self.current_token_tpl
        logging.debug("current_token_tpl=" + str(self.current_token_tpl))
        (self.area_x_indices, self.edge_index, self.edge_type) = \
            get_forwardinput_forelement(global_idx, sense_idx, self.grapharea_matrix, self.area_size, self.graph_dataobj)

        return ((self.area_x_indices, self.edge_index, self.edge_type), self.next_token_tpl)

    def __len__(self):
        if self.counter == 0:
            length_reader = SLC.read_split(self.split_name)
            logging.debug("Preliminary: reading the dataset to determine the number of samples")
            try:
                while True:
                    length_reader.__next__()
                    self.counter = self.counter + 1
            except StopIteration:
                return self.counter
        else:
            return self.counter
#####


### Auxiliary function:
### Getting the graph-input (x, edge_index, edge_type)
### Here I decide what is the starting token for a prediction. For now, it is "sense if present, else global"
def get_forwardinput_forelement(global_idx, sense_idx, grapharea_matrix, area_size, graph_dataobj):

    logging.debug("get_forwardinput_forelement: " + str(global_idx) + ' ; ' + str(sense_idx))
    if (sense_idx == -1):
        sourcenode_idx = global_idx
    else:
        sourcenode_idx = sense_idx
    nodes, edge_index, edge_type = AD.get_node_data(grapharea_matrix, sourcenode_idx, area_size)
    area_x_indices = nodes.to(torch.long).to(DEVICE)

    return (area_x_indices, edge_index, edge_type)