import torch
import GNN.SenseLabeledCorpus as SLC
import GNN.NumericalIndices as NI
import logging
from Utils import DEVICE
import Graph.Adjacencies as AD


# Auxiliary function: pad with -1s
def pad_tensor(tensor, target_shape):
    padded_t = torch.ones(size=target_shape) * -1
    if len(tensor.shape) <=1: # 1 row
        padded_t[0][0:len(tensor)] = tensor
    else: # 2 or more rows
        for i in range(tensor.shape[0]):
            padded_t[i,0:len(tensor[i])] = tensor[i]
    return padded_t


# When automatic batching is enabled, collate_fn is called with a list of data samples at each time.
# It is expected to collate the input samples into a batch for yielding from the data loader iterator.
# At the moment, I am just returning a list, because the input of different sizes in the rgcn layer call
# does not allow me to stack the tensors to do proper batching
def collate_fn(data):
    # # data: is a list of tuples, each with (x, edge_index, edge_type) tensors
    # max_areasize = 0
    # max_columns = 0
    #
    # for ((x, edge_index, edge_type), label_next_token_tpl) in data:
    #     if x.shape[0] > max_areasize:
    #         max_areasize = x.shape[0]
    #     if x.shape[1] > max_columns:
    #         max_columns = x.shape[1]
    #
    # padded_input_ls = []
    # labels_ls = []
    # target_shape = (max_areasize, max_columns)
    #
    # for ((x, edge_index, edge_type), label_next_token_tpl) in data:
    #     padded_x = pad_tensor(x, target_shape)
    #     padded_edge_index = pad_tensor(edge_index, target_shape)
    #     padded_edge_type = pad_tensor(edge_type, target_shape)
    #     labels_ls.append(torch.tensor(label_next_token_tpl, dtype=torch.int64))
    #     sample_input_matrix = torch.cat([padded_x, padded_edge_index, padded_edge_type], dim=1)
    #     padded_input_ls.append(sample_input_matrix)

    padded_input_ls = []
    labels_ls = []

    for ((x, edge_index, edge_type), label_next_token_tpl) in data:
        padded_input_ls.append((x, edge_index, edge_type))
        labels_ls.append(label_next_token_tpl)

    return padded_input_ls, labels_ls


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
        (self.area_x, self.edge_index, self.edge_type) = \
            get_forwardinput_forelement(global_idx, sense_idx, self.grapharea_matrix, self.area_size, self.graph_dataobj)

        return ((self.area_x, self.edge_index, self.edge_type), self.next_token_tpl)

    def __len__(self):
        if self.counter == 0:
            length_reader = SLC.read_split(self.split_name)
            logging.info("Preliminary: reading the dataset to determine the number of samples")
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
    nodes_ls, edge_index, edge_type = AD.get_node_data(grapharea_matrix, sourcenode_idx, area_size)
    node_indices = torch.Tensor(sorted(nodes_ls)).to(torch.int64).to(DEVICE)
    area_x = graph_dataobj.x.index_select(0, node_indices)

    return (area_x, edge_index, edge_type)