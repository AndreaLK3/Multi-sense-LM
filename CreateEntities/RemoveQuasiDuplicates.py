import logging
import pandas as pd
import Utils
import os
import nltk
import string
import re

NUM_INSTANCES_CHUNK = 1000
STOPWORDS_CORENLP_FILEPATH = os.path.join("CreateEntities",'stopwords_coreNLP.txt')


# Utility function
def get_wordnet_pos(word):
    """Map POS tag to the first character, that nltk.stemlemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": 'a',# wordnet.ADJ,
                "N": 'n',# wordnet.NOUN,
                "V": 'v',# wordnet.VERB,
                "R": 'r'}# wordnet.ADV}

    return tag_dict.get(tag, 'n')

# Removing punctuation. Removing stopwords. Then removing duplicates
def process_def_or_example(element_text, stopwords_ls):
    element_text_nopunct = re.sub("["+str(string.punctuation)+"]", " ", element_text)
    tokens = nltk.tokenize.word_tokenize(element_text_nopunct)
    tokens_nostopwords = list(filter(lambda tok: tok not in stopwords_ls, tokens))
    elem_newtext = ' '.join(tokens_nostopwords)
    return elem_newtext


# The objective is to  use stopwords removal and punctuation removal to eliminate quasi-duplicates
# – those elements that differ for a “to” or for a comma
# elements_name must be one of: 'definitions', 'examples'
def eliminate_duplicates_in_word(word, elements_name, extended_lang_id='english'):
    Utils.init_logging(os.path.join("CreateEntities","PreprocessInput.log"), logging.INFO)
    
    hdf5_input_filepath = os.path.join(Utils.FOLDER_INPUT, elements_name +".h5")
    hdf5_output_filepath = os.path.join(Utils.FOLDER_INPUT, Utils.PROCESSED + '_' +elements_name + ".h5")
    input_db = pd.HDFStore(hdf5_input_filepath, mode='r')

    stopwords_ls = nltk.corpus.stopwords.words(extended_lang_id)

    hdf5_min_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'bn_id': Utils.HDF5_BASE_SIZE_512 / 4,
                          Utils.DEFINITIONS: Utils.HDF5_BASE_SIZE_512 / 2, Utils.EXAMPLES: Utils.HDF5_BASE_SIZE_512 / 2}
    min_itemsize_dict = {key: hdf5_min_itemsizes[key] for key in ['word', 'bn_id', elements_name]}

    with pd.HDFStore(hdf5_output_filepath, mode='w') as outfile:

        word_df = input_db.select(key=elements_name, where="word == " + str(word))
        bn_ids = set(word_df['bn_id'])
        new_data = []

        for bn_id in bn_ids:
            sense_df = word_df.loc[word_df['bn_id'] == bn_id]
            # a tuple contains; bn_id, def/example, processed def/example
            processed_elements = list(map(lambda elem_text: process_def_or_example(elem_text, stopwords_ls),
                                          sense_df[elements_name]))
            sense_lts = list(zip(sense_df.bn_id, sense_df[elements_name], processed_elements))
            sense_lts_01_no_duplicates = []
            for tpl in sense_lts:
                if processed_elements.count(tpl[2]) < 2:
                    sense_lts_01_no_duplicates.append(tpl)
                else:
                    logging.info("Duplicate element found: " + str(tpl[1]))
                    processed_elements.remove(tpl[2])

            data_to_add = list(map(lambda tpl: (word, tpl[0], tpl[1]) , sense_lts_01_no_duplicates))
            new_data.extend(data_to_add)

        new_df = pd.DataFrame(data=new_data, columns=['word', 'bn_id', elements_name])
        outfile.append(key=elements_name, value=new_df, min_itemsize=min_itemsize_dict)



def main():
    vocabulary = ['plant', 'wide', 'move', 'light']
    logging.info("Eliminating quasi-duplicates from definitions...")
    for word in vocabulary:
        eliminate_duplicates_in_word(word, Utils.DEFINITIONS, extended_lang_id='english')
    logging.info("Eliminating quasi-duplicates from examples...")
    for word in vocabulary:
        eliminate_duplicates_in_word(word, Utils.EXAMPLES, extended_lang_id='english')

