import Filesystem as F
import Utils
import pandas as pd
import os
import NN.DataLoading as DL
import NN.Training as T
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import NN.Models.RNNs as RNNs
import tables
import NN.ExplorePredictions as EP
import NN.NumericalIndices as NI
from itertools import cycle
import sqlite3
import logging
import NN.MostFrequentSense as MFS



# Near-copy of what is found in NN\MostFrequentSense.py (test-driven development). More log, usinng MiniCorpus
def compute_MFS_for_corpus():
    Utils.init_logging("Tests-compute_MFS_for_corpus.log")
    # ---------- 1) Initializing folders ---------
    subfolder = F.FOLDER_SENSELABELED
    graph_folder = os.path.join(F.FOLDER_GRAPH, subfolder)
    inputdata_folder = os.path.join(F.FOLDER_INPUT, subfolder)
    vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, subfolder)
    folders = (graph_folder, inputdata_folder, vocabulary_folder)

    # ---------- 2) initialize the HDF5 archive for the MostFrequentSense, senses'DB, and the counts' dictionary ----------
    corpus_fpath = os.path.join(F.FOLDER_MINICORPUSES, F.FOLDER_SENSELABELED)
    mfs_archive_fpath = os.path.join(corpus_fpath, F.MOST_FREQ_SENSE_FILE)
    mfs_archive = pd.HDFStore(mfs_archive_fpath, mode='w')

    senseindices_db = sqlite3.connect(os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB))
    senseindices_db_c = senseindices_db.cursor()

    wIdx_mfsIdx_w_mfs_lts = []
    hdf5_min_itemsizes = {Utils.WORD + Utils.INDEX: Utils.HDF5_BASE_SIZE_512 / 4,
                          Utils.MOST_FREQUENT_SENSE + Utils.INDEX: Utils.HDF5_BASE_SIZE_512 / 4,
                          Utils.WORD: Utils.HDF5_BASE_SIZE_512 / 4,
                          Utils.MOST_FREQUENT_SENSE: Utils.HDF5_BASE_SIZE_512 / 4}
    counts_nested_dict = {}

    # ---------- 3) create a Reader for the sense-labeled corpus ----------
    # Setting up the graph, grapharea_matrix and vocabulary
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix = T.get_objects(True,folders)
    objects = graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix

    # Creating the iterators for training, validation and test datasets --------------------
    corpus_fpath = os.path.join(F.FOLDER_MINICORPUSES, subfolder) # The only modification in a MiniExperiment
    _model_forDataLoading = RNNs.RNN(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                           embeddings_matrix, include_globalnode_input=False,
                           batch_size=1 , n_layers=1, n_hid_units=1024)# unused, just needed for the loading function
    datasets, dataloaders = T.get_dataloaders(objects, True, corpus_fpath, folders,
                                              batch_size=1, seq_len=1, model_forDataLoading=_model_forDataLoading)

    train_dataset, _, _ = datasets
    train_dataloader, _, _ = dataloaders
    train_dataiter = iter(cycle(train_dataloader))

    next_token_tpl = None
    # read the text, get the global label and sense label
    for b_idx in range(len(train_dataloader)-1):
        batch_input, batch_labels = train_dataiter.__next__()

        batch_globals_indices = batch_input[:, :, 0] - train_dataset.last_sense_idx
        batch_senses_indices = batch_input[:, :, (batch_input.shape[2] // 2)]

        input_global_index = batch_globals_indices[0,0].item()
        input_word = EP.get_globalword_fromindex_df(input_global_index, vocabulary_folder)
        input_sense_index = batch_senses_indices[0,0].item()
        input_sense = MFS.get_sense_from_idx(senseindices_db_c, input_sense_index)

        logging.debug("(input_global_index, input_sense_index)=" + str((input_global_index, input_sense_index))
              + " ; (input_word, input_sense)=" + str((input_word, input_sense)))

        # keep a count of the sense label
        # word = vocabulary_df.iloc[input_global_index]['word']
        lemmatized_word = vocabulary_df.iloc[input_global_index]['lemmatized_form']
        logging.debug(lemmatized_word + "_" + str(input_sense_index))
        try:
            prev_freq = counts_nested_dict[lemmatized_word][input_sense_index]
            counts_nested_dict[lemmatized_word][input_sense_index] = prev_freq + 1
        except KeyError:
            counts_nested_dict[lemmatized_word] = {}
            counts_nested_dict[lemmatized_word][input_sense_index] = 1

    senseindices_x_words_df = pd.DataFrame.from_dict(counts_nested_dict)
    mfs_vocabulary_ls = senseindices_x_words_df.columns.tolist()
    for w_idx in range(len(mfs_vocabulary_ls)):
        lemmatized_word = mfs_vocabulary_ls[w_idx]
        try:
            mfs_idx = senseindices_x_words_df[lemmatized_word].idxmax()
            mfs = MFS.get_sense_from_idx(senseindices_db_c, mfs_idx)
            wIdx_mfsIdx_w_mfs_lts.append((w_idx, mfs_idx, lemmatized_word, mfs))
        except KeyError:
            logging.warning("computing MFS: Could not find key="+ str(lemmatized_word) +" in senseindices_x_words_df. Moving on")

    mfs_df_columns = [Utils.WORD + Utils.INDEX, Utils.MOST_FREQUENT_SENSE + Utils.INDEX,
                          Utils.WORD, Utils.MOST_FREQUENT_SENSE]
    mfs_df = pd.DataFrame(data=wIdx_mfsIdx_w_mfs_lts, columns=mfs_df_columns)

    # store in HDF5
    mfs_archive.append(key=Utils.MOST_FREQUENT_SENSE, value=mfs_df, min_itemsize=hdf5_min_itemsizes)
    tables.file._open_files.close_all()