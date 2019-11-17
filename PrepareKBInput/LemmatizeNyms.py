import logging
import pandas as pd
import Utils
import os
import nltk

NUM_INSTANCES_CHUNK = 1000
STOPWORDS_CORENLP_FILEPATH = os.path.join("PrepareKBInput",'stopwords_coreNLP.txt')

# Utility function
def get_wordnet_pos(word):
    """Map POS tag to the first character, that nltk.stemlemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
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
def lemmatize_nyms_in_word(word, elements_name, input_db, output_db):

    lemmatizer = nltk.stem.WordNetLemmatizer()

    hdf5_min_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'bn_id': Utils.HDF5_BASE_SIZE_512 / 4,
                          Utils.SYNONYMS: Utils.HDF5_BASE_SIZE_512 / 4, Utils.ANTONYMS: Utils.HDF5_BASE_SIZE_512 / 4}
    min_itemsize_dict = {key: hdf5_min_itemsizes[key] for key in ['word', 'bn_id', elements_name]}

    try:
        word_df = Utils.select_from_hdf5(input_db, elements_name, ["word"], [word])
    except KeyError: # no elements of that kind for the word (e.g. it has no antonyms)
        logging.info("Did not found any " + elements_name + " for word: " + word + ". Moving on")
        return
    bn_ids = set(word_df['bn_id'])
    new_data = []

    for bn_id in bn_ids:
        sense_df = word_df.loc[word_df['bn_id'] == bn_id]
        sense_lts = list(zip(sense_df.bn_id, sense_df[elements_name]))

        sense_lts_01 = list(map(
            lambda tpl: (tpl[0], lemmatize_term(tpl[1], lemmatizer)),
                        sense_lts))
        sense_lts_01_lemmatized = list(set(sense_lts_01))

        data_to_add = list(map(lambda tpl: (word, tpl[0], tpl[1]) , sense_lts_01_lemmatized))

        new_df = pd.DataFrame(data=data_to_add, columns=['word', 'bn_id', elements_name])
        output_db.append(key=elements_name, value=new_df, min_itemsize=min_itemsize_dict)

