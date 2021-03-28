import torch
import SenseLabeledCorpus as SLC
import Models.NumericalIndices as NI
import logging
from Utils import DEVICE
import Graph.Adjacencies as AD
import VocabularyAndEmbeddings.Vocabulary_Utilities as VU
import os
import Filesystem as F
import Utils
import io
import VocabularyAndEmbeddings.Vocabulary_Utilities as VocabUtils
import sqlite3

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
def standardtextcorpus_generator(txt_corpus_fpath):

    logging.info("setting up standardtextcorpus_generator on path: " + str(txt_corpus_fpath))
    with io.open(txt_corpus_fpath,'r', encoding="utf-8") as text_file:
        for i,line in enumerate(text_file):
            # FastText has vectors for '@-@', '@.@', but T-XL does not, so we convert them into '-', '.' and join
            line_tokens, _ = VocabUtils.process_line(line, 0)
            escaped_tokens = list(map(lambda word_token: VocabUtils.process_word_token({'surface_form':word_token}, lowercasing=False), line_tokens))
            for token in escaped_tokens:
                token_dict ={'surface_form':token}
                yield token_dict

##### The Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_fpath, corpus_type, inputdata_folder, vocab_df, model,
                       grapharea_matrix, area_size, graph_dataobj):
        self.textcorpus_path = corpus_fpath
        self.sensecorpus_or_text = corpus_type
        self.generator = SLC.read_split(corpus_fpath) if self.sensecorpus_or_text else standardtextcorpus_generator(corpus_fpath)
        self.vocab_df = vocab_df
        self.nn_model = model
        self.counter = 0
        self.globals_unk_counter = 0
        db_filepath = os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB)
        indicesTable_db = sqlite3.connect(db_filepath)
        self.senseindices_db_c = indicesTable_db.cursor()
        self.last_sense_idx = self.senseindices_db_c.execute("SELECT COUNT(*) from indices_table").fetchone()[0]
        self.first_idx_dummySenses = Utils.get_startpoint_dummySenses(inputdata_folder)

        self.grapharea_matrix = grapharea_matrix
        self.area_size = area_size
        self.graph_dataobj = graph_dataobj
        self.next_token_tpl = None

    def __getitem__(self, index):
        self.current_token_tpl, self.next_token_tpl = \
            NI.get_tokens_tpls(self.next_token_tpl, self.generator, self.senseindices_db_c, self.vocab_df,
                               self.grapharea_matrix, self.last_sense_idx, self.first_idx_dummySenses,
                               self.sensecorpus_or_text)

        global_idx, sense_idx = self.current_token_tpl
        #logging.info("self.current_token_tpl="+ str(self.current_token_tpl))
        #logging.info("TextDataset > global_idx, sense_idx =" + str((global_idx, sense_idx)))
        relative_global_idx = global_idx + self.nn_model.last_idx_senses
        (global_forwardinput_triple, sense_forwardinput_triple)= \
            get_forwardinput_forelement(relative_global_idx, sense_idx, self.grapharea_matrix, self.area_size)
        #logging.info("self.next_token_tpl=" + str(self.next_token_tpl))

        return ((global_forwardinput_triple, sense_forwardinput_triple), self.next_token_tpl)

    def __len__(self):
        logging.debug("Requesting the length of the dataset")
        if self.counter == 0:
            length_reader = standardtextcorpus_generator(self.textcorpus_path) \
                if self.generator.__name__ == 'standardtextcorpus_generator' else SLC.read_split(self.textcorpus_path)
            logging.info("Preliminary: reading the dataset to determine the number of samples")
            try:
                while True:
                    length_reader.__next__()
                    self.counter = self.counter + 1
            except StopIteration:
                return self.counter
        else:
            return self.counter

    def get_num_unks(self):
        unk_index = self.vocab_df[self.vocab_df['word'] == Utils.UNK_TOKEN].index.values[0]
        if self.globals_unk_counter == 0:
            unk_reader = standardtextcorpus_generator(self.textcorpus_path) \
                if self.generator.__name__ == 'standardtextcorpus_generator' else SLC.read_split(self.textcorpus_path)
            logging.info("Reading the dataset to determine the number of <unk> tokens among the globals")
            try:
                while True:
                    self.current_token_tpl, self.next_token_tpl = \
                        NI.get_tokens_tpls(self.next_token_tpl, unk_reader, self.senseindices_db_c, self.vocab_df,
                                           self.grapharea_matrix, self.last_sense_idx, self.first_idx_dummySenses,
                                           self.sensecorpus_or_text)

                    global_idx, sense_idx = self.current_token_tpl
                    if global_idx == unk_index:
                        self.globals_unk_counter = self.globals_unk_counter +1
                        if self.globals_unk_counter % 1000 == 0:
                            logging.info("self.globals_unk_counter=" + str(self.globals_unk_counter))
            except StopIteration:
                return self.globals_unk_counter
        else:
            return self.globals_unk_counter
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