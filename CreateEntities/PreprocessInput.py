import logging
import pandas as pd
import Utils
import os
import nltk
import string
import itertools

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

# We lemmatize synonyms and antonyms
def modify_synonym_or_antonym(nym, lemmatizer):
    nym_lemmatized = lemmatizer.lemmatize(nym, get_wordnet_pos(nym))
    return nym_lemmatized


# Removing punctuation. Removing stopwords. Then removing duplicates
def modify_def_or_example(element_text, stopwords_ls):
    tokens = nltk.tokenize.word_tokenize(element_text)
    tokens_nopunct = list(filter(lambda tok: tok not in string.punctuation, tokens))
    tokens_nostopwords = list(filter(lambda tok: tok not in stopwords_ls, tokens_nopunct))
    elem_newtext = ' '.join(tokens_nostopwords)
    return elem_newtext


# Depending on which kind of element we are processing,
def apply_modification(elements_name, element_text, stopwords_ls, lemmatizer):
    switch_dict = {
        Utils.DEFINITIONS : modify_def_or_example(element_text, stopwords_ls),
        Utils.EXAMPLES : modify_def_or_example(element_text, stopwords_ls),
        Utils.SYNONYMS : modify_synonym_or_antonym(element_text, lemmatizer),
        Utils.ANTONYMS: modify_synonym_or_antonym(element_text, lemmatizer)
    }

    return switch_dict.get(elements_name)


# elements_name must be one of: 'definitions', 'examples', 'synonyms', 'antonyms'
def preprocess_elements(vocabulary, elements_name, extended_lang_id='english'):
    Utils.init_logging(os.path.join("CreateEntities","PreprocessInput.log"), logging.INFO)
    
    hdf5_input_filepath = os.path.join(Utils.FOLDER_INPUT, elements_name +".h5")
    hdf5_output_filepath = os.path.join(Utils.FOLDER_INPUT, Utils.PROCESSED + '_' +elements_name + ".h5")
    input_db = pd.HDFStore(hdf5_input_filepath, mode='r')

    stopwords_ls = nltk.corpus.stopwords.words(extended_lang_id)
    # adding other sources for stopwords will be considered
    lemmatizer = nltk.stem.WordNetLemmatizer()

    hdf5_min_itemsizes = {'word': Utils.HDF5_BASE_CHARSIZE / 4, 'bn_id': Utils.HDF5_BASE_CHARSIZE / 4,
                               Utils.DEFINITIONS:Utils.HDF5_BASE_CHARSIZE/2, Utils.EXAMPLES:Utils.HDF5_BASE_CHARSIZE/2,
                               Utils.SYNONYMS:Utils.HDF5_BASE_CHARSIZE/4, Utils.ANTONYMS:Utils.HDF5_BASE_CHARSIZE/4}
    min_itemsize_dict = {key: hdf5_min_itemsizes[key] for key in ['word', 'bn_id', elements_name]}

    with pd.HDFStore(hdf5_output_filepath, mode='w') as outfile:
        for word in vocabulary:

            word_df = input_db.select(key=elements_name, where="word == " + str(word))
            bn_ids = set(word_df['bn_id'])
            new_data = []

            for bn_id in bn_ids:
                sense_df = word_df.loc[word_df['bn_id'] == bn_id]
                sense_lts = list(zip(sense_df.bn_id, sense_df[elements_name]))

                sense_lts_01 = list(map(
                    lambda tpl: (tpl[0], apply_modification(elements_name, tpl[1], stopwords_ls, lemmatizer)),
                                sense_lts))
                sense_lts_01_no_duplicates = list(set(sense_lts_01))

                data_toadd = list(map(lambda tpl: (word, tpl[0], tpl[1]) , sense_lts_01_no_duplicates))
                new_data.extend(data_toadd)

            new_df = pd.DataFrame(data=new_data, columns=['word', 'bn_id', elements_name])
            logging.info(new_df)
            outfile.append(key=elements_name, value=new_df, min_itemsize=min_itemsize_dict)



# current mini-vocabulary: ['plant', 'wide', 'move']

# preprocess_elements(vocabulary, elements_name, extended_lang_id='english')

