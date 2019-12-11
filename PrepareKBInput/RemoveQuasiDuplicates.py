import logging
import pandas as pd
import Utils
import os
import nltk
import string
import re
from itertools import cycle


STOPWORDS_CORENLP_FILEPATH = os.path.join("PrepareKBInput",'stopwords_coreNLP.txt')


# Removing punctuation. Removing stopwords. It will be used to remove duplicates
def process_def_or_example(element_text, stopwords_ls):
    element_text_nopunct = re.sub("["+str(string.punctuation)+"]", " ", element_text)
    tokens = nltk.tokenize.word_tokenize(element_text_nopunct)
    tokens_nostopwords = list(filter(lambda tok: tok not in stopwords_ls, tokens))
    elem_newtext = ' '.join(tokens_nostopwords)
    return elem_newtext


# The objective is to use stopwords removal and punctuation removal to eliminate quasi-duplicates
# – those elements that differ for a “to” or for a comma
# elements_name must be one of: 'definitions', 'examples'
def eliminate_duplicates_in_table(vocabulary_ls, elements_name, input_db, output_db, extended_lang_id='english'):

    stopwords_ls = nltk.corpus.stopwords.words(extended_lang_id)

    hdf5_min_itemsizes = {Utils.SENSE_WN_ID: Utils.HDF5_BASE_SIZE_512 / 4,
                          Utils.DEFINITIONS: Utils.HDF5_BASE_SIZE_512 , Utils.EXAMPLES: Utils.HDF5_BASE_SIZE_512 }
    min_itemsize_dict = {key: hdf5_min_itemsizes[key] for key in [Utils.SENSE_WN_ID, elements_name]}

    all_word_senses = list(set(input_db[elements_name][Utils.SENSE_WN_ID]))
    word_senses_toprocess = sorted([sense_str for sense_str in all_word_senses
                                    if Utils.get_word_from_sense(sense_str) in vocabulary_ls])
    new_data = []

    for wn_id in word_senses_toprocess:
        sense_df = Utils.select_from_hdf5(input_db, elements_name, [Utils.SENSE_WN_ID], [wn_id])
        # a tuple contains; bn_id, def/example, processed def/example
        processed_elements = list(map(lambda elem_text: process_def_or_example(elem_text, stopwords_ls),
                                      sense_df[elements_name]))
        sense_lts = list(zip(sense_df[elements_name], processed_elements))
        sense_lts_01_no_duplicates = []
        for tpl in sense_lts:
            if processed_elements.count(tpl[1]) < 2:
                sense_lts_01_no_duplicates.append(tpl)
            else:
                logging.info("Duplicate element (" + elements_name+ ") found for '" +wn_id+ "' : " + str(tpl[1]))
                logging.info(sense_lts)
                logging.info('***')
                processed_elements.remove(tpl[1])

        data_to_add = list(map(lambda tpl: (wn_id, tpl[0]) , sense_lts_01_no_duplicates))
        new_data.extend(data_to_add)

    new_df = pd.DataFrame(data=new_data, columns=[Utils.SENSE_WN_ID, elements_name])
    output_db.append(key=elements_name, value=new_df, min_itemsize=min_itemsize_dict)



