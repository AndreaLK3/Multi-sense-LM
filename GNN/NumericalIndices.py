import logging
import sqlite3
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import WordNetError
import Utils
from Vocabulary import Vocabulary_Utilities as VocabUtils
from time import time

### Utility function, to handle the common labeling mistake: 'next%3:00...' should be 'next%5:00...'
def try_to_get_wordnet_sense(wn30_key):
    try:
        wordnet_sense = wn.lemma_from_key(wn30_key).synset().name()
    except (WordNetError, ValueError):
        try:
            wn30_key = wn30_key.replace('3', '5', 1)
            wordnet_sense = wn.lemma_from_key(wn30_key).synset().name()
        except (WordNetError, ValueError):
            wordnet_sense = None
    return wordnet_sense


### Internal function to: translate the word (and if present, the sense) into numerical indices.
# sense = [0,se) ; single prototype = [se,se+sp) ; definitions = [se+sp, se+sp+d) ; examples = [se+sp+d, e==num_nodes)
def convert_tokendict_to_tpl(token_dict, senseindices_db_c, globals_vocabulary_h5):

    keys = token_dict.keys()
    sense_index_queryresult = None

    if 'wn30_key' in keys:

        wn30_key = token_dict['wn30_key']
        wordnet_sense = try_to_get_wordnet_sense(wn30_key)
        if wordnet_sense is not None:
            try:
                query = "SELECT vocab_index FROM indices_table " + "WHERE word_sense='" + wordnet_sense + "'"
                sense_index_queryresult = senseindices_db_c.execute(query).fetchone()
            except sqlite3.OperationalError :
                logging.info("Error while attempting to execute query: " + query + " . Skipping sense")

        if sense_index_queryresult is None: # there was no sense-key, or we did not find the sense for the key
            sense_index = -1
        else:
            sense_index = sense_index_queryresult[0]
    else:
        sense_index = -1
    word = VocabUtils.process_word_token(token_dict) # html.unescape

    try:
        global_absolute_index = Utils.select_from_hdf5(globals_vocabulary_h5, 'vocabulary', ['word'], [word]).index[0]
    except IndexError: # redirect onto <unk>
        word = Utils.UNK_TOKEN
        global_absolute_index = Utils.select_from_hdf5(globals_vocabulary_h5, 'vocabulary', ['word'], [word]).index[0]

    global_index = global_absolute_index # + last_idx_senses; do not add this to globals, or we go beyond the n_classes
    # we still have to add the relative displacement of last_idx_senses to the global, but not here

    logging.debug('(global_index, sense_index)=' + str((global_index, sense_index)))
    return (global_index, sense_index)

### Entry point function to: translate the word (and if present, the sense) into numerical indices.
def get_tokens_tpls(next_token_tpl, split_datagenerator, senseindices_db_c, vocab_h5):
    if next_token_tpl is None:
        current_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),
                                                     senseindices_db_c, vocab_h5)
        next_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),
                                                  senseindices_db_c, vocab_h5)
    else:
        current_token_tpl = next_token_tpl
        next_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),senseindices_db_c, vocab_h5)

    return current_token_tpl, next_token_tpl