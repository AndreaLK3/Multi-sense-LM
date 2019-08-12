import Utils
import logging
import WordNet
import BabelNet
import DBpedia
import Wiktionary
import OmegaWiki
import pandas as pd
import os
import time

NUM_WORDS_IN_FILE = 5000
HDF5_BASE_CHARSIZE = 1024


def refine_synonyms(synonyms_ls, target_word, exclude_multiword=True):
    space_characters = ['_', ' ']
    final_synonyms = [s for s in synonyms_ls if s != target_word]

    if exclude_multiword:
        for syn in synonyms_ls:
            oneword_flag = True
            for c in space_characters:
                if c in syn:
                    oneword_flag = False
            if oneword_flag:
                final_synonyms.append(syn)


    return final_synonyms


def getAndSave_inputData(vocabulary=[]):
    Utils.init_logging("RetrieveInputData.log", logging.INFO)
    if not(os.path.exists(Utils.FOLDER_RAW_INPUT)):
        os.mkdir(Utils.FOLDER_RAW_INPUT)

    toy_vocabulary = ['light']#["wide", "plant", "sea", "high", "move"]
    vocabulary = toy_vocabulary


    # Note: the words of the vocabulary must be processed in alphabetic order, to guarantee that the data in the
    # tables in the HDF5 storage is sorted.
    vocabulary_sorted = sorted(vocabulary)

    tasks = [WordNet.retrieve_DESA, Wiktionary.retrieve_DE, OmegaWiki.retrieve_DS,
             DBpedia.retrieve_dbpedia_def, BabelNet.retrieve_DES]
    sources = [Utils.SOURCE_WORDNET, Utils.SOURCE_WIKTIONARY, Utils.SOURCE_OMEGAWIKI, Utils.SOURCE_DBPEDIA, Utils.SOURCE_BABELNET]

    categories = [Utils.DEFINITIONS, Utils.EXAMPLES, Utils.SYNONYMS, Utils.ANTONYMS, Utils.ENCYCLOPEDIA_DEF]
    hdf5_min_itemsizes_dict = {Utils.DEFINITIONS:HDF5_BASE_CHARSIZE, Utils.EXAMPLES:HDF5_BASE_CHARSIZE,
                               Utils.SYNONYMS:HDF5_BASE_CHARSIZE, Utils.ANTONYMS:HDF5_BASE_CHARSIZE,
                               Utils.ENCYCLOPEDIA_DEF:4*HDF5_BASE_CHARSIZE}

    storage_filenames = [Utils.H5_raw_defs, Utils.H5_examples, Utils.H5_synonyms, Utils.H5_antonyms, Utils.H5_enc_defs]
    storage_filepaths = list(map(lambda fn: os.path.join(Utils.FOLDER_RAW_INPUT, fn) , storage_filenames))
    open_storage_files = [ pd.HDFStore(fname, mode='w') for fname in storage_filepaths] #reset HDF5 archives

    categories_returned = [[0,1,2,3],[0,1],[0,2],[4], [0,1,2]]
    num_columns = [4,2,2,1, 3]

    for i in range(len(tasks)):
        task = tasks[i]
        source = sources[i]
        cats = categories_returned[i]
        n_cols = num_columns[i]

        for word in vocabulary_sorted:
            task_time_start = time.clock()
            desa = task(word)
            task_runtime = time.clock() - task_time_start
            logging.info("Word: " + word + " ; source: " + source + " ; task runtime = " + str(round(task_runtime,5)))
            logging.debug(desa)
            for j in range(n_cols):
                current_column = categories[cats[j]]
                logging.info(current_column)
                # Post-process if needed. No multi-word synonyms for now
                if current_column == Utils.SYNONYMS:
                    column_data = refine_synonyms(desa[j], word)
                else:
                    column_data = desa[j]
                df = pd.DataFrame(data=column_data, columns=[current_column])

                (open_storage_files[cats[j]]).append(key=current_column, value=df,
                                                     min_itemsize={key:hdf5_min_itemsizes_dict[key] for key in [current_column]})

    for storagefile in open_storage_files:
        storagefile.close()

