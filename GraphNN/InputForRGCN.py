import logging
import sqlite3
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import WordNetError
import Utils
from Vocabulary import Vocabulary_Utilities as VocabUtils
from time import time

### Internal function to: translate the word (and if present, the sense) into numerical indices.
# sense = [0,se) ; single prototype = [se,se+sp) ; definitions = [se+sp, se+sp+d) ; examples = [se+sp+d, e==num_nodes)
def convert_tokendict_to_tpl(token_dict, senseindices_db_c, globals_vocabulary_h5, last_idx_senses):
    keys = token_dict.keys()
    sense_index_queryresult = None

    if 'wn30_key' in keys:
        try:
            try:
                wn30_key = token_dict['wn30_key']
                wordnet_sense = wn.lemma_from_key(wn30_key).synset().name()
            except WordNetError: # common labeling mistake: 'next%3:00...' should be 'next%5:00...'
                wn30_key = (token_dict['wn30_key']).replace('3', '5')
                wordnet_sense = wn.lemma_from_key(wn30_key).synset().name()
            query = "SELECT vocab_index FROM indices_table " + "WHERE word_sense='" + wordnet_sense + "'"
            sense_index_queryresult = senseindices_db_c.execute(query).fetchone()
        except ValueError: # it may fail, due to typo or wrong labeling
            logging.info("Did not find word sense for key = " + token_dict['wn30_key'])
        except sqlite3.OperationalError :
            logging.info("Error while attempting to execute query: " + query + " . Skipping sense")

        if sense_index_queryresult is None: # the was no sense-key, or we did not find the sense for the key
            sense_index = -1
        else:
            sense_index = sense_index_queryresult[0]
    else:
        sense_index = -1
    word = VocabUtils.process_slc_token(token_dict)
    try:
        global_absolute_index = Utils.select_from_hdf5(globals_vocabulary_h5, 'vocabulary', ['word'], [word]).index[0]
    except IndexError:
        # global_absolute_index = Utils.select_from_hdf5(globals_vocabulary_h5, 'vocabulary', ['word'], [Utils.UNK_TOKEN]).index[0]
        raise Utils.MustSkipUNK_Exception

    global_index = global_absolute_index # + last_idx_senses; do not add this to globals, or we go beyond the n_classes
    logging.debug('(global_index, sense_index)=' + str((global_index, sense_index)))
    return (global_index, sense_index)

### Entry point function to: translate the word (and if present, the sense) into numerical indices.
def get_tokens_tpls(next_token_tpl, split_datagenerator, senseindices_db_c, vocab_h5, last_idx_senses):
    if next_token_tpl is None:
        current_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),
                                                     senseindices_db_c, vocab_h5, last_idx_senses)
        next_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),
                                                  senseindices_db_c, vocab_h5, last_idx_senses)
    else:
        current_token_tpl = next_token_tpl
        next_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(),senseindices_db_c, vocab_h5, last_idx_senses)
    return current_token_tpl, next_token_tpl