import logging
import pandas as pd
import Utils
import os
import nltk
from itertools import cycle

NUM_INSTANCES_CHUNK = 1000
STOPWORDS_CORENLP_FILEPATH = os.path.join("PrepareKBInput",'stopwords_coreNLP.txt')

# Utility function
def get_wordnet_pos(word):
    """Map POS tag to the first character, that nltk.stemlemmatize() accepts"""
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
    except TypeError as e:
        logging.warning("word=" + word)
        logging.warning(e)
        tag = 'n'
    tag_dict = {"J": 'a',# wordnet.ADJ,
                "N": 'n',# wordnet.NOUN,
                "V": 'v',# wordnet.VERB,
                "R": 'r'}# wordnet.ADV}

    return tag_dict.get(tag, 'n')

# We lemmatize synonyms and antonyms
def lemmatize_term(nym, lemmatizer):
    nym_lemmatized = lemmatizer.lemmatize(nym, get_wordnet_pos(nym))
    return nym_lemmatized


# elements_name must be one of: 'synonyms', 'antonyms'
def lemmatize_nyms_in_word(vocabulary_ls, elements_name, input_db, output_db):

    lemmatizer = nltk.stem.WordNetLemmatizer()

    hdf5_min_itemsizes = {Utils.SENSE_WN_ID: Utils.HDF5_BASE_SIZE_512 / 4,
                          Utils.SYNONYMS: Utils.HDF5_BASE_SIZE_512 / 4, Utils.ANTONYMS: Utils.HDF5_BASE_SIZE_512 / 4}
    min_itemsize_dict = {key: hdf5_min_itemsizes[key] for key in [Utils.SENSE_WN_ID, elements_name]}

    all_word_senses = list(set(input_db[elements_name][Utils.SENSE_WN_ID]))
    word_senses_toprocess = sorted([sense_str for sense_str in all_word_senses if
                             Utils.get_word_from_sense(sense_str) in vocabulary_ls])
    data_to_add = []

    for wn_id in word_senses_toprocess:
        sense_df = Utils.select_from_hdf5(input_db, elements_name, [Utils.SENSE_WN_ID], [wn_id])
        sense_lts = list(zip(cycle([wn_id]), sense_df[elements_name]))

        sense_lts_lemmatized = list(map(
            lambda tpl: (tpl[0], lemmatize_term(tpl[1], lemmatizer)),
                        sense_lts))
        sense_lts_lemmatized_noduplicates = list(set(sense_lts_lemmatized))

        data_to_add.extend(list(map(lambda tpl: (tpl[0], tpl[1]) , sense_lts_lemmatized_noduplicates)))

    new_df = pd.DataFrame(data=data_to_add, columns=[Utils.SENSE_WN_ID, elements_name])
    output_db.append(key=elements_name, value=new_df, min_itemsize=min_itemsize_dict)

