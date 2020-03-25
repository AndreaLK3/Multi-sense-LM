import torch
import copy
import GNN.SenseLabeledCorpus as SLC
import GNN.NumericalIndices as NI
import logging
from Utils import DEVICE
import Graph.Adjacencies as AD
import Vocabulary.Vocabulary_Utilities as VU
import os
import Filesystem as F
import Utils

# Auxiliary function to pack an input tuple (x_indices, edge_index, edge_type)
# into a tensor [x_indices; edge_sources; edge_destinations; edge_type]
def pack_input_tuple_into_tensor(input_tuple, graph_area):

    max_edges = int(graph_area**1.5)
    in_tensor = - 1 * torch.ones(size=(graph_area + max_edges*3,)).to(torch.long)
    x_indices = input_tuple[0]
    edge_sources = input_tuple[1][0]
    edge_destinations = input_tuple[1][1]
    edge_type = input_tuple[2]
    if len(edge_sources) > max_edges:
        logging.warning("Num edges=" + str(len(edge_sources)) + " , while max_edges packed=" + str(max_edges))
    in_tensor[0:len(x_indices)] = x_indices
    in_tensor[graph_area: graph_area+min(len(edge_sources), max_edges)] = edge_sources[0:max_edges]
    in_tensor[graph_area+max_edges:graph_area+max_edges+min(len(edge_destinations), max_edges)] = edge_destinations[0:max_edges]
    in_tensor[graph_area+2*max_edges:graph_area+2*max_edges+min(len(edge_type), max_edges)] = edge_type[0:max_edges]
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
        globals_input_ls = []
        senses_input_ls = []
        for ((global_input_tpl, sense_input_tpl), label_next_token_tpl) in data:
            if i >= self.sequence_length:
                globals_reunited = torch.stack(globals_input_ls, dim=0)
                senses_reunited = torch.stack(senses_input_ls, dim=0)
                input_lls.append(torch.cat([globals_reunited, senses_reunited], dim=1))
                i=0
                globals_input_ls = []
                senses_input_ls = []

            globals_input_ls.append(pack_input_tuple_into_tensor(global_input_tpl, self.grapharea_size))
            senses_input_ls.append(pack_input_tuple_into_tensor(sense_input_tpl, self.grapharea_size))
            i = i + 1
            labels_ls.append(torch.tensor(label_next_token_tpl).to(torch.int64).to(DEVICE))
        # add the last one
        globals_reunited = torch.stack(globals_input_ls, dim=0)
        senses_reunited = torch.stack(senses_input_ls, dim=0)
        input_lls.append(torch.cat([globals_reunited, senses_reunited], dim=1))
        return torch.stack(input_lls, dim=0), torch.stack(labels_ls, dim=0)


##### Auxiliary function: reading a standard text corpus into the dataset,
##### without the need to use the SLC facility to process a sense-labeled XML
def standardtextcorpus_generator(split_name):
    if split_name==Utils.TRAINING:
        folder = F.FOLDER_TRAIN
    elif split_name==Utils.VALIDATION:
        folder = F.FOLDER_VALIDATION
    else: # Utils.TEST
        folder = F.FOLDER_TEST

    in_folder_path = os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_MYTEXTCORPUS, folder)
    logging.info("setting up standardtextcorpus_generator on path: " + str(in_folder_path))
    textfiles_fnames = os.listdir(in_folder_path)
    with [open(os.path.join(in_folder_path, fname),'r') for fname in textfiles_fnames][0] as text_file:
        for i, line in enumerate(text_file):
            if line == '':
                break
            tokens_in_line_truecase, _tot_tokens = VU.process_line(line, tot_tokens=0)
            for token in tokens_in_line_truecase:
                token_dict ={'surface_form':token} # to use the same refinement as the tokens from sense-labeled corpus
                yield token_dict

##### The Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, sensecorpus_or_text, split_name, senseindices_db_c, vocab_h5, model,
                       grapharea_matrix, area_size, graph_dataobj):
        self.split_name = split_name
        self.generator = SLC.read_split(split_name) if sensecorpus_or_text else standardtextcorpus_generator(split_name)
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
        (global_forwardinput_triple, sense_forwardinput_triple)= \
            get_forwardinput_forelement(global_idx, sense_idx, self.grapharea_matrix, self.area_size)

        return ((global_forwardinput_triple, sense_forwardinput_triple), self.next_token_tpl)

    def __len__(self):
        if self.counter == 0:
            length_reader = standardtextcorpus_generator(self.split_name) \
                if self.generator.__name__ == 'standardtextcorpus_generator' else SLC.read_split(self.split_name)
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
### Here I decide what is the input for a prediction. It is going to be (global, sense[-1s if not present])
def get_forwardinput_forelement(global_idx, sense_idx, grapharea_matrix, area_size):

    logging.debug("get_forwardinput_forelement: " + str(global_idx) + ' ; ' + str(sense_idx))
    area_x_indices_global, edge_index_global, edge_type_global = AD.get_node_data(grapharea_matrix, global_idx, area_size)
    if (sense_idx == -1):
        area_x_indices_sense = torch.zeros(size=(area_x_indices_global.shape)).to(DEVICE)
        edge_index_sense = torch.zeros(size=(edge_index_global.shape)).to(DEVICE)
        edge_type_sense = torch.zeros(size=(edge_type_global.shape)).to(DEVICE)
    else:
        area_x_indices_sense, edge_index_sense, edge_type_sense = AD.get_node_data(grapharea_matrix, sense_idx,
                                                                                      area_size)

    return ( (area_x_indices_global, edge_index_global, edge_type_global),
             (area_x_indices_sense, edge_index_sense, edge_type_sense))