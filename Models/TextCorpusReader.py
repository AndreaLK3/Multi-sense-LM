import pandas as pd
import torch

import Graph.PolysemousWords
import Utils
import Filesystem as F
import VocabularyAndEmbeddings.Vocabulary as V
import VocabularyAndEmbeddings.Vocabulary_Utilities as VocabUtils
import os
import logging
import numpy as np
import SenseLabeledCorpus as SLC
from Graph import DefineGraph as DG, Adjacencies as AD
from Models.DataLoading import DataLoading as DL
from Utils import DEVICE
from VocabularyAndEmbeddings.ComputeEmbeddings import SpMethod

# Auxiliary function:
# Input: corpus and split
# Outcome: get the filepaths of the text file and the numerical pre-encoding
def get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls):

    if corpus_name.lower() == F.WT2.lower():
        if split == Utils.TRAINING:
            split_fname = F.WT_TRAIN_FILE
        elif split == Utils.VALIDATION:
            split_fname = F.WT_VALID_FILE
        elif split == Utils.TEST:
            split_fname = F.WT_TEST_FILE
        txt_corpus_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT2], split_fname)
        numIDs_outfile_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT2], split_fname + F.CORPUS_NUMERICAL_EXTENSION
                                            + "withVocabFrom_" + "_".join(vocabulary_sources_ls) + ".npy")

    if corpus_name.lower() == F.SEMCOR.lower():
        if split == Utils.TRAINING:
            split_dirname = F.FOLDER_TRAIN
        elif split == Utils.VALIDATION:
            split_dirname = F.FOLDER_VALIDATION
        elif split == Utils.TEST:
            split_dirname = F.FOLDER_TEST
        txt_corpus_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], split_dirname)
        numIDs_outfile_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], split_dirname + F.CORPUS_NUMERICAL_EXTENSION
                                            + "withVocabFrom_" + "_".join(vocabulary_sources_ls) + ".npy")
    return txt_corpus_fpath, numIDs_outfile_fpath


# Auxiliary function:
# Input: Corpus name, split, sources used to create the vocabulary that we wish to use
# Outcome: load the numpy pre-encoded data
def load_corpus_IDs(corpus_name, split, vocabulary_sources_ls):
    txt_corpus_fpath, numIDs_outfile_fpath = get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls)
    numerical_IDs_t = np.load(numIDs_outfile_fpath)
    logging.info("Loaded the encoded corpus at " + str(numIDs_outfile_fpath))
    return numerical_IDs_t

# Auxiliary function: Setting up the graph, grapharea_matrix (used for speed) and the vocabulary
def get_objects(vocab_sources_ls, sp_method=Utils.SpMethod.FASTTEXT, grapharea_size=32):

    graph_folder, inputdata_folder, vocabulary_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    graph_dataobj = DG.get_graph_dataobject(False, vocab_sources_ls, sp_method).to(DEVICE)

    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, grapharea_size, hops_in_area=1, graph_folder=graph_folder)

    embeddings_matrix = torch.tensor(np.load(os.path.join(inputdata_folder, F.SPVs_FILENAME))).to(torch.float32)

    globals_vocabulary_fpath = os.path.join(vocabulary_folder, "vocabulary.h5")
    vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')

    vocabulary_numSensesList = vocabulary_df['num_senses'].to_list().copy()
    if all([num_senses == -1 for num_senses in vocabulary_numSensesList]):
        vocabulary_df = Graph.PolysemousWords.compute_globals_numsenses(graph_dataobj, grapharea_matrix, grapharea_size,
                                                                        inputdata_folder, globals_vocabulary_fpath)

    return graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix, inputdata_folder


# Top-level function:
# Read a corpus from a text file, convert the tokens into their numerical IDs, save the result
# The conversion relies on a vocabulary made from the specified sources, with default min_count values
def encode_txt_corpus(corpus_name, split, vocabulary_sources_ls, lowercase=False):
    Utils.init_logging("TextCorpusReader-read_txt_corpus.log")

    txt_corpus_fpath, numIDs_outfile_fpath = get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls)
    vocab_df = V.get_vocabulary_df(vocabulary_sources_ls, lowercase, txt_min_count=1, slc_min_count=2)
    words_ls = list(vocab_df["word"])
    logging.info("Length of the list of words from the vocabulary: " + str(len(words_ls)))
    numerical_IDs_ls = []

    tot_tokens=0
    with open(txt_corpus_fpath, "r", encoding="utf-8") as txt_corpus_file:
        for i,line in enumerate(txt_corpus_file):
            # words = line.split() + ['<eos>']
            line_tokens, tot_tokens = VocabUtils.process_line(line, tot_tokens)
            line_tokens = list(map(lambda word_token:
                                   VocabUtils.process_word_token({'surface_form':word_token}, lowercase), line_tokens))
            line_tokens_IDs = list(map(lambda tok: words_ls.index(tok), line_tokens))
            logging.debug("line_tokens_IDs=" + str(line_tokens_IDs))
            numerical_IDs_ls.extend(line_tokens_IDs)
            if (i%500) == 0:
                logging.info("Reading and encoding the corpus. " + str(tot_tokens) + ' tokens processed...')

    numerical_IDs_arr = np.array(numerical_IDs_ls)
    np.save(numIDs_outfile_fpath, numerical_IDs_arr)

# Top-level function:
# Read a sense-labeled corpus in UFSAC format, convert the tokens into their numerical IDs, save the result
# ****** Currently unused, because reading in SemCor does not require only (global_idx, sense_idx) but also
# ****** ((area_x_indices_global, edge_index_global, edge_type_global), (area_x_indices_sense, edge_index_sense, edge_type_sense))
# def encode_slc_corpus(corpus_name, split, vocabulary_sources_ls=[F.WT2, F.SEMCOR]):
#     slc_dir_fpath, numIDs_outfile_fpath = get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls)
#     sp_method = SpMethod.FASTTEXT
#     gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocabulary_sources_ls, sp_method)
#
#     objects = get_objects(vocabulary_sources_ls, sp_method=SpMethod.FASTTEXT, grapharea_size=32)
#     dataset, dataloader = setup_corpus(objects, slc_dir_fpath, slc_or_text=True, gr_in_voc_folders= gr_in_voc_folders,
#                                        batch_size=1, seq_len=1)
#     corpus_lts = []
#     while True:
#         try:
#             global_input_triple, sense_input_triple = dataset.__getitem__(0)
#             current_global = global_input_triple[0][0]
#             current_sense = sense_input_triple[0][0]
#             corpus_lts.append((current_global, current_sense))
#         except StopIteration:
#             break
#
#     numerical_IDs_arr = np.ndarray(corpus_lts)
#     np.save(numIDs_outfile_fpath, numerical_IDs_arr)


# Top-level function: get dataset and dataloader on a corpus, specifying filepath and type (slc vs. text)
def setup_corpus(objects, corpus_location, slc_or_text, gr_in_voc_folders, batch_size, seq_len):
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix, inputdata_folder = objects
    graph_folder, inputdata_folder, vocabulary_folder = gr_in_voc_folders

    bptt_collator = DL.BPTTBatchCollator(grapharea_size, seq_len)
    dataset = DL.TextDataset(corpus_location, slc_or_text, inputdata_folder, vocabulary_df,
                             grapharea_matrix, grapharea_size, graph_dataobj)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size * seq_len,
                                                   num_workers=0, collate_fn=bptt_collator)

    return dataset, dataloader