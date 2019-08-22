import logging
import pandas as pd
import Utils
import os
import nltk
import string
import re

NUM_INSTANCES_CHUNK = 1000
STOPWORDS_CORENLP_FILENAME = 'stopwords_coreNLP.txt'


# elements_name must be either 'definitions' or 'examples'
def preprocess_elements(elements_name='examples', extended_lang_id='english'):
    Utils.init_logging("PreprocessInput.log", logging.INFO)
    hdf5_input_filepath = os.path.join(Utils.FOLDER_INPUT, elements_name +".h5")
    hdf5_output_filepath = os.path.join(Utils.FOLDER_INPUT, "preprocessed_"+elements_name +".h5")
    df_chunksIterator = pd.read_hdf(hdf5_input_filepath, mode='r',iterator=True, chunksize=NUM_INSTANCES_CHUNK)

    stopwords_ls = nltk.corpus.stopwords.words(extended_lang_id)
    # adding other sources for stopwords will be considered

    #we also remove all non-hyphen punctuation
    puncts_nohyphen = string.punctuation.replace('-', '')
    puncts_nohyphen_pattern_str = '['+puncts_nohyphen+']'

    hdf5_min_itemsizes_dict = {'word': HDF5_BASE_CHARSIZE / 4, 'source': HDF5_BASE_CHARSIZE / 4,
                               Utils.DEFINITIONS: HDF5_BASE_CHARSIZE, Utils.EXAMPLES: HDF5_BASE_CHARSIZE,
                               Utils.SYNONYMS: HDF5_BASE_CHARSIZE / 2, Utils.ANTONYMS: HDF5_BASE_CHARSIZE / 2,
                               Utils.ENCYCLOPEDIA_DEF: 4 * HDF5_BASE_CHARSIZE}

    with pd.HDFStore(hdf5_output_filepath, mode='w') as outfile:

        # the columns are: (index, word, examples/definitions, source)
        for chunk in df_chunksIterator:
            for row in chunk.itertuples():
                logging.info(row.word)
                logging.info(row[2])
                elem_nopuncts = re.sub(puncts_nohyphen_pattern_str, '', row[2])
                logging.info(elem_nopuncts)
                elem_alltokens = nltk.tokenize.word_tokenize(elem_nopuncts.lower())
                elem_tokens = list(filter(lambda t: t not in stopwords_ls, elem_alltokens))
                logging.info(elem_tokens)



