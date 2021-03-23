import Filesystem as F
import Utils
import pandas as pd
import os
import Models.DataLoading as DL
import Models.Training as T
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import Models.Variants.RNNs as RNNs
import tables
import Models.ExplorePredictions as EP
import Models.NumericalIndices as NI
from itertools import cycle
import sqlite3
import logging
import Models.ComputeMFS as MFS
import SenseLabeledCorpus as SLC
from scipy import sparse
import numpy as np
from time import time

def compute_MFS_for_corpus():
    t0 = time()
    Utils.init_logging("Tests-compute_MFS_for_corpus.log")
    # Init
    subfolder = F.FOLDER_SENSELABELED
    graph_folder = os.path.join(F.FOLDER_GRAPH, subfolder)
    inputdata_folder = os.path.join(F.FOLDER_INPUT, subfolder)
    vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, subfolder)
    corpus_folder = os.path.join(F.FOLDER_MYTESTS, F.FOLDER_MINICORPUSES, F.FOLDER_SENSELABELED)
    folders = (graph_folder, inputdata_folder, vocabulary_folder)

    # More init, necessary objects
    generator = SLC.read_split(os.path.join(corpus_folder, F.FOLDER_TRAIN))

    objects = T.get_objects(True, folders)
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix = objects
    _model_forDataLoading = RNNs.RNN(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                           embeddings_matrix, include_globalnode_input=False,
                           batch_size=1 , n_layers=1, n_hid_units=1024)# unused, just needed for the loading function
    datasets, _dataloaders = T.get_dataloaders(objects, True, corpus_folder, folders,
                                              batch_size=1, seq_len=1, model_forDataLoading=_model_forDataLoading)
    training_dataset = datasets[0]

    vocab_ls = training_dataset.vocab_df['word'].to_list().copy()
    lemm_words_x_senses_mat = np.zeros(shape=(len(vocab_ls), training_dataset.last_sense_idx))

    # Iterate over the sense corpus
    for xml_instance in generator:
        (global_index, sense_index) = \
            NI.convert_tokendict_to_tpl(xml_instance, training_dataset.senseindices_db_c, training_dataset.vocab_df,
                                                     training_dataset.grapharea_matrix, training_dataset.last_sense_idx,
                                                     training_dataset.first_idx_dummySenses, slc_or_text=True)
        lemmatized_word = training_dataset.vocab_df.iloc[global_index]['lemmatized_form']
        lemmatized_idx = vocab_ls.index(lemmatized_word)
        sense = MFS.get_sense_from_idx(training_dataset.senseindices_db_c, sense_index)
        logging.info(str((lemmatized_word, sense)))
        lemm_words_x_senses_mat[lemmatized_idx, sense_index] = lemm_words_x_senses_mat[global_index, sense_index]+1

    # Iterate over the vocabulary (ends up only on lemmatized words' rows), compute the max, gather the mfs data
    wIdx_mfsIdx_w_mfs_lts = []
    for row_i in range(lemm_words_x_senses_mat.shape[0]):
        mat_wordrow = lemm_words_x_senses_mat[row_i]
        if np.count_nonzero(mat_wordrow):
            mfs_idx =  np.argmax(mat_wordrow)
            # logging.info(str((row_i,mfs_idx)))

            lemmatized_word = vocab_ls[row_i]
            mfs = MFS.get_sense_from_idx(training_dataset.senseindices_db_c, mfs_idx)
            wIdx_mfsIdx_w_mfs_lts.append((row_i, mfs_idx, lemmatized_word, mfs))

    # create Pandas dataframe
    mfs_df_columns = [Utils.WORD + Utils.INDEX, Utils.MOST_FREQUENT_SENSE + Utils.INDEX,
                  Utils.WORD, Utils.MOST_FREQUENT_SENSE]
    mfs_df = pd.DataFrame(data=wIdx_mfsIdx_w_mfs_lts, columns=mfs_df_columns)

    # create HDF5, store in HDF5
    hdf5_min_itemsizes = {Utils.WORD + Utils.INDEX: Utils.HDF5_BASE_SIZE_512 / 4,
                      Utils.MOST_FREQUENT_SENSE + Utils.INDEX: Utils.HDF5_BASE_SIZE_512 / 4,
                      Utils.WORD: Utils.HDF5_BASE_SIZE_512 / 4,
                      Utils.MOST_FREQUENT_SENSE: Utils.HDF5_BASE_SIZE_512 / 4}
    mfs_archive_fpath = os.path.join(corpus_folder, 'bis_'+F.MOST_FREQ_SENSE_FILE)
    mfs_archive = pd.HDFStore(mfs_archive_fpath, mode='w')
    mfs_archive.append(key=Utils.MOST_FREQUENT_SENSE, value=mfs_df, min_itemsize=hdf5_min_itemsizes)

    t1 = time()
    Utils.log_chronometer([t0,t1])
    tables.file._open_files.close_all()
