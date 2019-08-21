import Utils
import logging
import WordNet
import BabelNet
import DBpedia
import Wiktionary
import OmegaWiki
import pandas as pd
import os
from itertools import cycle
import re

NUM_WORDS_IN_FILE = 5000
HDF5_BASE_CHARSIZE = 1024
CATEGORIES = [Utils.DEFINITIONS, Utils.EXAMPLES, Utils.SYNONYMS, Utils.ANTONYMS, Utils.ENCYCLOPEDIA_DEF]

def refine_nyms(nyms_ls, target_word, exclude_multiword=True):
    oneword_nyms = []

    space_characters = ['_', ' ']
    nyms_ls = [s for s in nyms_ls if s != target_word]


    if exclude_multiword:
        for syn in nyms_ls:
            oneword_flag = True
            for c in space_characters:
                if c in syn:
                    oneword_flag = False
            if oneword_flag:
                oneword_nyms.append(syn)

    final_nyms = [oneword_nyms[i] for i in range(len(oneword_nyms)) if oneword_nyms[i] not in oneword_nyms[i+1:]]

    return final_nyms


def getWordData(word, sources, tasks_info_dict, open_storage_files, hdf5_min_itemsizes_dict, lang_id):
    synonyms_already_inserted = []
    antonyms_already_inserted = []
    for source in sources:
        info = tasks_info_dict[source]

        task = info['task']
        cats = info['categories']
        n_cols = info['n_cols']

        desa = task(word)
        for j in range(n_cols):
            current_column = CATEGORIES[cats[j]]
            logging.info(current_column)
            # Post-process: No target word. No duplicates. No multi-word synonyms for now
            if current_column in [Utils.SYNONYMS, Utils.ANTONYMS]:
                if current_column == Utils.SYNONYMS:
                    source_synonyms = refine_nyms(desa[j], word)
                    logging.debug("source_synonyms : " + str(source_synonyms))
                    synonyms_to_add = list(set(source_synonyms).difference(synonyms_already_inserted))
                    logging.debug("synonyms_to_add : " + str(synonyms_to_add))
                    column_data = synonyms_to_add
                    synonyms_already_inserted.extend(synonyms_to_add)
                else:
                    source_antonyms = refine_nyms(desa[j], word)
                    antonyms_to_add = list(set(source_antonyms).difference(antonyms_already_inserted))
                    column_data = antonyms_to_add
                    antonyms_already_inserted.extend(antonyms_to_add)
            else:
                column_data = desa[j]
            column_data_targetLanguage = list(filter(lambda elem: Utils.check_language(elem, lang_id), column_data))
            logging.debug("column_data_targetLanguage : " + str(column_data_targetLanguage))
            df_data = zip(cycle([word]), column_data_targetLanguage, cycle([source]))
            df_columns = ['word', current_column, 'source']
            df = pd.DataFrame(data=df_data, columns=df_columns)

            (open_storage_files[cats[j]]).append(key=current_column, value=df,
                                                 min_itemsize={key: hdf5_min_itemsizes_dict[key] for key in df_columns})



def getAndSave_inputData(vocabulary=[], use_mini_vocabulary=True, lang_id='en'):
    Utils.init_logging("RetrieveInputData.log", logging.INFO)
    if not(os.path.exists(Utils.FOLDER_INPUT)):
        os.mkdir(Utils.FOLDER_INPUT)

    if use_mini_vocabulary:
        vocabulary = ["high"]#,"wide", 'low', "plant",  "move", "sea", 'light', ]

    # Note: the words of the vocabulary must be processed in alphabetic order, to guarantee that the data in the
    # tables in the HDF5 storage is sorted.
    vocabulary_sorted = sorted(vocabulary)

    tasks = [WordNet.retrieve_DESA, Wiktionary.retrieve_DESA, OmegaWiki.retrieve_DS,
             DBpedia.retrieve_dbpedia_def, BabelNet.retrieve_DES]
    sources = [Utils.SOURCE_WORDNET, Utils.SOURCE_WIKTIONARY, Utils.SOURCE_OMEGAWIKI, Utils.SOURCE_DBPEDIA, Utils.SOURCE_BABELNET]
    categories_returned = [[0,1,2,3],[0,1,2,3],[0,2],[4], [0,1,2]]
    num_columns = [4,4,2,1,3]

    tasks_info_dict = {}
    for i in range(len(sources)):
        tasks_info_dict[sources[i]]={'task':tasks[i], 'categories':categories_returned[i],'n_cols':num_columns[i]}


    hdf5_min_itemsizes_dict = {'word': HDF5_BASE_CHARSIZE/4, 'source':HDF5_BASE_CHARSIZE/4,
                               Utils.DEFINITIONS:HDF5_BASE_CHARSIZE, Utils.EXAMPLES:HDF5_BASE_CHARSIZE,
                               Utils.SYNONYMS:HDF5_BASE_CHARSIZE/2, Utils.ANTONYMS:HDF5_BASE_CHARSIZE/2,
                               Utils.ENCYCLOPEDIA_DEF:4*HDF5_BASE_CHARSIZE}

    storage_filenames = [categ + ".h5" for categ in CATEGORIES]
    storage_filepaths = list(map(lambda fn: os.path.join(Utils.FOLDER_INPUT, fn), storage_filenames))
    open_storage_files = [ pd.HDFStore(fname, mode='w') for fname in storage_filepaths] #reset HDF5 archives


    for word in vocabulary_sorted:
        getWordData(word, sources, tasks_info_dict, open_storage_files, hdf5_min_itemsizes_dict, lang_id)


    for storagefile in open_storage_files:
        storagefile.close()
