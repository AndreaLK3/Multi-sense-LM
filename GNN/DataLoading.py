import torch
import GNN.SenseLabeledCorpus as SLC
import GNN.NumericalIndices as NI
import logging
from Utils import DEVICE
import Graph.Adjacencies as AD


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
        self.grahp_dataobj = graph_dataobj
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



### Step n.2: getting the graph-input (x, edge_index, edge_type)
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