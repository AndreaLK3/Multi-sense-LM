import Filesystem as F
import Utils
import pandas as pd
import os
import Models.TrainingSetup as TS
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import Models.Variants.RNNs as RNNs
import tables
import Models.ExplorePredictions as EP
import Models.NumericalIndices as NI
from itertools import cycle
import sqlite3
import logging
import SenseLabeledCorpus as SLC
from time import time
import numpy as np

def get_sense_from_idx(senseindices_db_c, sense_index):
    senseindices_db_c.execute("SELECT word_sense FROM indices_table WHERE vocab_index=" + str(sense_index))
    sense_name_row = senseindices_db_c.fetchone()
    if sense_name_row is not None:
        sense_name = sense_name_row[0]
    else:
        sense_name = None
    return sense_name

def compute_MFS_for_corpus(vocab_sources_ls=[F.WT2, F.SEMCOR], sp_method=CE.Method.FASTTEXT):
    t0 = time()

    # Init
    graph_folder, inputdata_folder, vocabulary_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    corpus_trainsplit_folder = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], F.FOLDER_TRAIN)  # always the SLC corpus
    folders = (graph_folder, inputdata_folder, vocabulary_folder)

    generator = SLC.read_split(corpus_trainsplit_folder)
    objects = TS.get_objects(vocab_sources_ls, sp_method, grapharea_size=32)
    _, model_forDataLoading, _ = TS.setup_model(model_type=TS.ModelType.RNN, include_globalnode_input=0, use_gold_lm=False,
            K=0, load_saved_model=False, sp_method=sp_method, context_method=None, C=0, dim_qkv=0, grapharea_size=32,
            batch_size=1, vocab_sources_ls=vocab_sources_ls) # unused, needed only for the loading function

    training_dataset, _ = TS.setup_corpus(objects, corpus_trainsplit_folder, True, folders, batch_size=32, seq_len=35,
                                          model_forDataLoading = model_forDataLoading)

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
        #sense = get_sense_from_idx(training_dataset.senseindices_db_c, sense_index)
        # logging.info(str((lemmatized_word, sense)))
        lemm_words_x_senses_mat[lemmatized_idx, sense_index] = lemm_words_x_senses_mat[global_index, sense_index]+1

    # Iterate over the vocabulary (ends up only on lemmatized words' rows), compute the max, gather the mfs data
    wIdx_mfsIdx_w_mfs_lts = []
    for row_i in range(lemm_words_x_senses_mat.shape[0]):
        mat_wordrow = lemm_words_x_senses_mat[row_i]
        if np.count_nonzero(mat_wordrow):
            mfs_idx =  np.argmax(mat_wordrow)
            # logging.info(str((row_i,mfs_idx)))

            lemmatized_word = vocab_ls[row_i]
            mfs = get_sense_from_idx(training_dataset.senseindices_db_c, mfs_idx)
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
    mfs_archive_fpath = F.MOST_FREQ_SENSE_FPATH
    mfs_archive = pd.HDFStore(mfs_archive_fpath, mode='w')
    mfs_archive.append(key=Utils.MOST_FREQUENT_SENSE, value=mfs_df, min_itemsize=hdf5_min_itemsizes)

    t1 = time()
    Utils.log_chronometer([t0,t1])
    tables.file._open_files.close_all()