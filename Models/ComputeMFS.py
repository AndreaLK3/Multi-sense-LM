import Filesystem as F
import Utils
import pandas as pd
import os
import Models.ExplorePredictions as EP
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import VocabularyAndEmbeddings.Vocabulary as V
import tables
import Graph.DefineGraph as DG
import Graph.Adjacencies as AD
import Models.NumericalIndices as NI
from itertools import cycle
import sqlite3
import logging
import SenseLabeledCorpus as SLC
from time import time
import numpy as np

# Auxiliary function
def get_sense_from_idx(senseindices_db_c, sense_index):
    senseindices_db_c.execute("SELECT word_sense FROM indices_table WHERE vocab_index=" + str(sense_index))
    sense_name_row = senseindices_db_c.fetchone()
    if sense_name_row is not None:
        sense_name = sense_name_row[0]
    else:
        sense_name = None
    return sense_name


# Auxiliary function: given a (sub) dictionary that contains keys=sense_indices and values_frequency,
# return the sense_index with the maximum frequency
def get_sense_with_max_frequency(senses_freq_dict):
    max_freq = 0
    most_frequent_sense = None

    for sense_idx in senses_freq_dict.keys():
        freq = senses_freq_dict[sense_idx]
        if freq > max_freq:
            max_freq = freq
            most_frequent_sense = sense_idx

    return most_frequent_sense, max_freq


def compute_MFS_for_corpus(vocab_sources_ls=[F.WT2, F.SEMCOR], sp_method=Utils.SpMethod.FASTTEXT):
    t0 = time()

    # ----- Initialization -----
    # Folders
    graph_folder, inputdata_folder, vocabulary_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    corpus_trainsplit_folder = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], F.FOLDER_TRAIN)  # always the SLC corpus
    # Vocabulary
    vocab_df = V.get_vocabulary_df(vocab_sources_ls, lowercase=False)
    vocab_ls = vocab_df["word"].to_list()
    # Reader
    generator = SLC.read_split(corpus_trainsplit_folder)
    # Senseindices_db
    senseindices_db_filepath = os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB)
    senseindices_db = sqlite3.connect(senseindices_db_filepath)
    senseindices_db_c = senseindices_db.cursor()
    # Grapharea_matrix
    graph_dataobj = DG.get_graph_dataobject(False, vocab_sources_ls, sp_method)
    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, area_size=32, hops_in_area=1, graph_folder=graph_folder, new=False)
    # Node indices
    last_sense_idx = senseindices_db_c.execute("SELECT COUNT(*) from indices_table").fetchone()[0]
    first_idx_dummySenses = Utils.get_startpoint_dummySenses(inputdata_folder)

    # ----- Counting the senses' frequency -----
    # Setting up the nested dictionary: word_index -> sense_index -> frequency
    word_senses_frequency_dict = dict()
    # Iterating over the sense corpus
    for xml_instance in generator:
        (global_index, sense_index) = \
            NI.convert_tokendict_to_tpl(xml_instance, senseindices_db_c, vocab_df,
                                                     grapharea_matrix, last_sense_idx,
                                                     first_idx_dummySenses, slc_or_text=True)
        try:
            senses_frequency_dict = word_senses_frequency_dict[global_index]
        except KeyError:
            word_senses_frequency_dict[global_index] = {sense_index : 1}
            continue
        try:
            frequency = senses_frequency_dict[sense_index]
        except KeyError:
            senses_frequency_dict[sense_index] = 1
            continue
        word_senses_frequency_dict[global_index][sense_index] = frequency+1

    # ----- Iterate over the vocabulary: compute the max, gather the Most Frequent Sense data -----
    word_indices = word_senses_frequency_dict.keys()
    wIdx_mfsIdx_w_mfs_lts = []
    for word_idx in word_indices:
        senses_freq_dict = word_senses_frequency_dict[word_idx]
        mfs_idx, freq = get_sense_with_max_frequency(senses_freq_dict)
        word = vocab_ls[word_idx]
        mfs = EP.get_sense_fromindex(mfs_idx, inputdata_folder)
        # Pack into a list of tuples
        wIdx_mfsIdx_w_mfs_lts.append((word_idx, mfs_idx, word, mfs))

    # ----- Insert in Dataframe, and save in HDF5 archive -----

    # create Pandas dataframe
    mfs_df_columns = [Utils.WORD + Utils.INDEX, Utils.MOST_FREQUENT_SENSE + Utils.INDEX,
                  Utils.WORD, Utils.MOST_FREQUENT_SENSE]
    mfs_df = pd.DataFrame(data=wIdx_mfsIdx_w_mfs_lts, columns=mfs_df_columns)

    # create HDF5, store in HDF5
    hdf5_min_itemsizes = {Utils.WORD + Utils.INDEX: Utils.HDF5_BASE_SIZE_512 / 4,
                      Utils.MOST_FREQUENT_SENSE + Utils.INDEX: Utils.HDF5_BASE_SIZE_512 / 4,
                      Utils.WORD: Utils.HDF5_BASE_SIZE_512 / 4,
                      Utils.MOST_FREQUENT_SENSE: Utils.HDF5_BASE_SIZE_512 / 4}
    mfs_archive_fpath = os.path.join(F.MFS_H5_FPATH)
    mfs_archive = pd.HDFStore(mfs_archive_fpath, mode='w')
    mfs_archive.append(key=Utils.MOST_FREQUENT_SENSE, value=mfs_df, min_itemsize=hdf5_min_itemsizes)

    t1 = time()
    Utils.log_chronometer([t0,t1])
    tables.file._open_files.close_all()