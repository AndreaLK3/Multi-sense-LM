import logging
import sqlite3
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import WordNetError
import Utils
from VocabularyAndEmbeddings import Vocabulary_Utilities as VocabUtils
from time import time
import Graph.Adjacencies as AD

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


### We use the current global' senseChildren edge in the graph, to pick a dummySense as the label.
def get_missing_sense_label(global_absolute_index, grapharea_matrix, last_sense_idx, first_idx_dummySenses):
    global_relative_index = global_absolute_index + last_sense_idx
    adjacent_nodes, edges, edge_type = AD.get_node_data(grapharea_matrix, global_relative_index, grapharea_size=32,
                                                     features_mask=(True, False, False))
    senses = [n for n in adjacent_nodes if n < last_sense_idx]
    dummySenses = [n for n in senses if first_idx_dummySenses < n]
    # if we are here, it's because the word had no specified sense label. Generally, it's because "for", "of" etc.
    # need a dummySense. In case of the occasional missing label (e.g. "was" does not have "be.v.01"), we pick the
    # first available sense
    if len(dummySenses) > 0:
        return dummySenses[0].item()  # (we can have only 1 here)
    else:
        try:
            return senses[0].item()
        except IndexError:
            logging.info("Could not find sense for: global_absolute_index=" + str(global_absolute_index))
            return -1


# Auxiliary function 1 for convert_tokendict_to_tpl: convert the standard word into a numerical index
def convert_globalword_to_idx(token_dict, slc_or_text, globals_vocabulary_df):
    word = VocabUtils.process_word_token(token_dict, lowercasing=slc_or_text)  # html.unescape + currently lowercasing
    # logging.info("convert_tokendict_to_tpl>word=" + word)
    word_row_df = globals_vocabulary_df[globals_vocabulary_df['word']==word]
    try:
        global_absolute_index = word_row_df.index.values[0]
    except IndexError: # redirect onto <unk>
        word = Utils.UNK_TOKEN
        word_row_df = globals_vocabulary_df[globals_vocabulary_df['word']==word]
        global_absolute_index = word_row_df.index.values[0]

    global_index = global_absolute_index # + last_idx_senses; do not add this to globals, or we go beyond the n_classes
    # we still have to add the relative displacement of last_idx_senses to the global, but not here
    return global_index, word_row_df

# Auxiliary function 2 for convert_tokendict_to_tpl: convert the sense into a numerical index
def convert_sense_to_idx(token_dict, globals_vocabulary_df, word_row_df, senseindices_db_c, grapharea_matrix,
                         last_sense_idx, first_idx_dummySenses, slc_or_text):
    # we lemmatize to avoid flunking inflected forms, like "irregularities"
    lemmatized_form = word_row_df.lemmatized_form.values[0]
    lemmatized_form_row_df = globals_vocabulary_df[globals_vocabulary_df['word']==lemmatized_form]
    lemmatized_form_index = lemmatized_form_row_df.index.values[0]

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
            sense_index = get_missing_sense_label(lemmatized_form_index, grapharea_matrix, last_sense_idx, first_idx_dummySenses) # -1
        else:
            sense_index = sense_index_queryresult[0]
    else:
        sense_index = get_missing_sense_label(lemmatized_form_index, grapharea_matrix, last_sense_idx, first_idx_dummySenses)
    return sense_index

### Internal function to: translate the word (and if present, the sense) into numerical indices.
# sense = [0,se) ; single prototype = [se,se+sp) ; definitions = [se+sp, se+sp+d) ; examples = [se+sp+d, e==num_nodes)
def convert_tokendict_to_tpl(token_dict, senseindices_db_c, vocab_df, grapharea_matrix, last_sense_idx,
                             first_idx_dummySenses, slc_or_text):

    global_index, h5_word_row = convert_globalword_to_idx(token_dict, slc_or_text, vocab_df)

    sense_index = convert_sense_to_idx(token_dict, vocab_df, h5_word_row, senseindices_db_c, grapharea_matrix,
                    last_sense_idx, first_idx_dummySenses, slc_or_text)

    logging.debug('(global_index, sense_index)=' + str((global_index, sense_index)))
    return (global_index, sense_index)

### Entry point function to: translate the word (and if present, the sense) into numerical indices.
def get_tokens_tpls(next_token_tpl, split_datagenerator, senseindices_db_c, vocab_df, grapharea_matrix, last_idx_sense,
                    first_idx_dummySenses, slc_or_text):
    if next_token_tpl is None:
        current_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(), senseindices_db_c, vocab_df,
                                                     grapharea_matrix, last_idx_sense, first_idx_dummySenses,
                                                     slc_or_text)
        next_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(), senseindices_db_c, vocab_df,
                                                  grapharea_matrix, last_idx_sense, first_idx_dummySenses, slc_or_text)
    else:
        current_token_tpl = next_token_tpl
        next_token_tpl = convert_tokendict_to_tpl(split_datagenerator.__next__(), senseindices_db_c, vocab_df,
                                                  grapharea_matrix, last_idx_sense, first_idx_dummySenses, slc_or_text)

    return current_token_tpl, next_token_tpl