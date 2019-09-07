import logging
import pandas as pd
import Utils
import os
import nltk
import string
import re

NUM_INSTANCES_CHUNK = 1000
STOPWORDS_CORENLP_FILEPATH = os.path.join("CreateEntities",'stopwords_coreNLP.txt')


# elements_name must be either 'definitions' or 'examples'
def preprocess_elements(elements_name, extended_lang_id):
    Utils.init_logging(os.path.join("CreateEntities","PreprocessInput.log"), logging.INFO)
    
    hdf5_input_filepath = os.path.join(Utils.FOLDER_INPUT, elements_name +".h5")
    hdf5_output_filepath = os.path.join(Utils.FOLDER_INPUT, "preprocessed_"+elements_name +".h5")
    df_chunksIterator = pd.read_hdf(hdf5_input_filepath, mode='r',iterator=True, chunksize=NUM_INSTANCES_CHUNK)

    stopwords_ls = nltk.corpus.stopwords.words(extended_lang_id)
    # adding other sources for stopwords will be considered

    #we also remove all non-hyphen punctuation
    puncts_nohyphen = string.punctuation.replace('-', '')
    puncts_nohyphen_pattern_str = '[' + puncts_nohyphen + ']'

    hdf5_min_itemsizes = {'word': Utils.HDF5_BASE_CHARSIZE / 4, 'source': Utils.HDF5_BASE_CHARSIZE / 4,
                               Utils.DEFINITIONS:20, Utils.EXAMPLES:20}
    min_itemsize_dict = {key: hdf5_min_itemsizes[key] for key in ['word', 'source', elements_name]}

    with pd.HDFStore(hdf5_output_filepath, mode='w') as outfile:

        # the columns are: (index, word, bn_id, examples/definitions, source)
        for chunk in df_chunksIterator:
            word_elem_source_lts = []
            for row in chunk.itertuples():
                logging.debug(row.word)
                elem_nopuncts = re.sub(puncts_nohyphen_pattern_str, ' ', row[2]) #may also remove entirely. see: n't VS 's
                elem_alltokens = nltk.tokenize.word_tokenize(elem_nopuncts.lower())
                elem_tokens = list(filter(lambda t: t not in stopwords_ls, elem_alltokens))
                logging.debug(elem_tokens)
                elem_string = ' '.join(elem_tokens)
                word_elem_source_lts.append((row.word, elem_string, row.source))

            df = pd.DataFrame(data=word_elem_source_lts, columns=['word',elements_name,'source'])
            logging.debug(df)
            outfile.append(key=elements_name, value=df, min_itemsize=min_itemsize_dict)


def preprocess_examples(extended_lang_id='english'):
    preprocess_elements(Utils.EXAMPLES, extended_lang_id)

def preprocess_definitions(extended_lang_id='english'):
    preprocess_elements(Utils.DEFINITIONS, extended_lang_id)