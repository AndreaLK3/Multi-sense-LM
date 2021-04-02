import logging
import os
import sqlite3

import nltk
import pandas as pd

import Filesystem as F
import SenseLabeledCorpus as SLC
import Utils
from GetKBInputData.LemmatizeNyms import lemmatize_term
from Models.NumericalIndices import try_to_get_wordnet_sense
from VocabularyAndEmbeddings import Vocabulary_Utilities as VocabUtils


def log_edges_minmax_nodes(edges_ls, name):
    if len(edges_ls) > 0:
        logging.info("Min source node in edges-" + name + " = " + str(min([tpl[0] for tpl in edges_ls])))
        logging.info("Max source node in edges-" + name + " = " + str(max([tpl[0] for tpl in edges_ls])))
        logging.info("Min target node in edges-" + name + " = " + str(min([tpl[1] for tpl in edges_ls])))
        logging.info("Max target node in edges-" + name + " = " + str(max([tpl[1] for tpl in edges_ls])))
    else:
        logging.info("len(edges_ls)==0")

# definitions -> senses : [se+sp, se+sp+d) -> [0,se)
# examples --> senses : [se+sp+d, e==num_nodes) -> [0,se)
def get_edges_elements(elements_name, elements_start_index_toadd, inputdata_folder):
    db_filepath = os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()

    indicesTable_db_c.execute("SELECT * FROM indices_table")
    edges_ls = []
    edges_toadd_counter = 0
    while (True):
        db_row = indicesTable_db_c.fetchone()
        if db_row is None:
            break
        target_idx = db_row[1]
        if elements_name==Utils.DEFINITIONS:
            start_sources = db_row[2] + elements_start_index_toadd
            end_sources = db_row[3] + elements_start_index_toadd
        else: # if elements_name==Utils.EXAMPLES:
            start_sources = db_row[4] + elements_start_index_toadd
            end_sources = db_row[5] + elements_start_index_toadd
        edges_toadd_counter = edges_toadd_counter + (end_sources-start_sources)
        if edges_toadd_counter > 0:
            for source in range(start_sources, end_sources):
                edges_ls.append((source, target_idx))
                edges_ls.append((target_idx, source))

    indicesTable_db.close()
    log_edges_minmax_nodes(edges_ls, elements_name)
    return edges_ls


# global -> senses : [se,se+sp) -> [0,se)
def get_edges_sensechildren(globals_voc_df, globals_start_index_toadd, inputdata_folder):

    db_filepath = os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()
    indicesTable_db_c.execute("SELECT * FROM indices_table")

    edges_ls = []

    while (True):
        db_row = indicesTable_db_c.fetchone()
        if db_row is None:
            break

        word_sense = db_row[0]
        word = Utils.get_word_from_sense(word_sense)
        logging.debug(word)
        sourceglobal_raw_idx = globals_voc_df.loc[globals_voc_df['word'] == word].index[0]
        sourceglobal_idx = globals_start_index_toadd + sourceglobal_raw_idx
        targetsense_idx = db_row[1]

        edges_ls.append((sourceglobal_idx, targetsense_idx))

    log_edges_minmax_nodes(edges_ls, "get_edges_sensechildren")
    indicesTable_db.close()
    return edges_ls


def get_additional_edges_sensechildren_from_slc(globals_voc_df, globals_start_index_toadd, inputdata_folder):
    logging.info("Reading the sense-labeled corpus, to create the connections between globals"
                 " and the senses that belong to other words.")
    train_corpus_fpath = os.path.join(F.FOLDER_TEXT_CORPORA, F.FOLDER_SENSELABELED, F.FOLDER_SEMCOR, Utils.TRAINING)
    slc_train_corpus_gen = SLC.read_split(train_corpus_fpath)
    senseindices_db = sqlite3.connect(os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB))
    senseindices_db_c = senseindices_db.cursor()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    edges_to_add_ls = []

    on_senselabeled = F.FOLDER_SENSELABELED in inputdata_folder  # to decide lowercasing

    try:
        while True:
            token_dict = slc_train_corpus_gen.__next__()
            keys = token_dict.keys()
            sense_index_queryresult = None

            # 1) Get the sense (and its index) specified in the SLC for the current token
            if 'wn30_key' in keys:
                wn30_key = token_dict['wn30_key']
                wordnet_sense = try_to_get_wordnet_sense(wn30_key)
                if wordnet_sense is not None:
                    try:
                        query = "SELECT vocab_index FROM indices_table " + "WHERE word_sense='" + wordnet_sense + "'"
                        sense_index_queryresult = senseindices_db_c.execute(query).fetchone()
                    except sqlite3.OperationalError:
                        logging.info("Error while attempting to execute query: " + query + " . Skipping sense")

                if sense_index_queryresult is None:  # there was no sense-key, or we did not find the sense for the key
                    continue # do nothing, we do not add a global-sense connection
                else:
                    targetsense_idx = sense_index_queryresult[0]
            else:
                continue # there was no sense-key specified for this token
            # 2) Get the global word of this token
            word = VocabUtils.process_word_token(token_dict, lowercasing=on_senselabeled)  # html.unescape
            lemmatized_word = lemmatize_term(word, lemmatizer)# since currently we always lemmatize in SelectK and other sense architectures
            if lemmatized_word not in wordnet_sense: # we are connecting all the "external" senses, e.g. say->state.v.01
                try:
                    sourceglobal_absolute_idx = globals_voc_df.loc[globals_voc_df['word'] == lemmatized_word].index[0]
                    global_relative_X_index = globals_start_index_toadd + sourceglobal_absolute_idx
                    edges_to_add_ls.append((global_relative_X_index, targetsense_idx))
                    #words_and_senses_ls.append((lemmatized_word, wordnet_sense))# for debug purposes
                except IndexError:  # global not present. No need to redirect onto <unk>, we skip
                    pass
            # else, we do not connect again the internal senses, e.g. say->say.v.01, we did that already
    except StopIteration:
        pass
    # remove duplicates
    edges_to_add_ls = list(set(edges_to_add_ls))
    log_edges_minmax_nodes(edges_to_add_ls, "get_additional_edges_sensechildren_from_slc")
    return edges_to_add_ls

# Edges: from globals to globals, inflected and parent form ('said' <--> 'say'). Type [4]
def get_edges_lemmatized(globals_vocabulary_df, globals_vocabulary_ls, last_sense_idx):
    edges_lemma_ls = []
    for tpl in globals_vocabulary_df.itertuples():
        word = tpl.word
        lemmatized_form = tpl.lemmatized_form
        if word == lemmatized_form:
            continue
        word_idx = globals_vocabulary_ls.index(word) + last_sense_idx
        lemmatized_idx = globals_vocabulary_ls.index(lemmatized_form) + last_sense_idx
        logging.info("Adding a Lemma edge between " + str(word_idx)+ "=" + word +
                     " and " + str(lemmatized_idx) + "=" + lemmatized_form)
        edges_lemma_ls.append((word_idx, lemmatized_idx))
        edges_lemma_ls.append((lemmatized_idx, word_idx))
    return edges_lemma_ls


def get_edges_selfloops(sc_edges, lemma_edges, num_globals, num_senses):

    sc_globals_sources = list(map(lambda edge_tpl : edge_tpl[0], sc_edges))
    lemma_globals_sources = list(map(lambda edge_tpl : edge_tpl[0], lemma_edges))
    lemma_globals_targets = list(map(lambda edge_tpl : edge_tpl[1], lemma_edges))
    globals_with_edges = sc_globals_sources + lemma_globals_sources + lemma_globals_targets

    all_globals_indices = list(range(num_senses,num_senses+num_globals))
    globals_needing_selfloop = [g_idx for g_idx in all_globals_indices if g_idx not in globals_with_edges]

    globals_edges_selfloops = [(g_idx, g_idx) for g_idx in globals_needing_selfloop]
    logging.info("globals_needing_selfloop=" + str(globals_needing_selfloop))
    # The inflected forms should be connected to their parent / lemmatized form
    # Therefore, this should yield no new edges

    #log_edges_minmax_nodes(globals_edges_selfloops, "get_edges_selfloops")
    return globals_edges_selfloops


# Synonyms and antonyms: global -> global : [se,se+sp) -> [se,se+sp).
# Bidirectional (which means 2 connections, (a,b) and (b,a)
def get_edges_nyms(nyms_name, globals_voc_df, globals_start_index_toadd, inputdata_folder):
    nyms_archive_fname = Utils.PROCESSED + '_' + nyms_name + '.h5'
    nyms_archive_fpath = os.path.join(inputdata_folder, nyms_archive_fname)

    nyms_df = pd.read_hdf(nyms_archive_fpath, key=nyms_name, mode="r")
    edges_ls = []
    on_senselabeled = F.FOLDER_SENSELABELED in inputdata_folder # to decide lowercasing
    logging.info("on_senselabeled=" + str(on_senselabeled))

    counter = 0
    for tpl in nyms_df.itertuples():
        word_sense = tpl.sense_wn_id
        word1 = Utils.get_word_from_sense(word_sense)
        word1 = VocabUtils.process_word_token({'surface_form': word1}, lowercasing=on_senselabeled)
        try:
            global_raw_idx_1 = globals_voc_df.loc[globals_voc_df['word'] == word1].index[0]
        except IndexError:
            logging.debug("Edges>" + nyms_name + ". Word '" + word1 + "' not found in globals' vocabulary. Skipping...")
            continue
        global_idx_1 = globals_start_index_toadd + global_raw_idx_1

        word2 = getattr(tpl, nyms_name)
        word2 = VocabUtils.process_word_token({'surface_form': word2}, lowercasing=on_senselabeled)
        if word1 == word2:
            continue # e.g. to avoid "friday" --> "Friday"
        try:
            global_raw_idx_2 = globals_voc_df.loc[globals_voc_df['word'] == word2].index[0]
        except IndexError:
            logging.debug("Edges>" + nyms_name + ". Word '" + word2 + "' not found in globals' vocabulary. Skipping...")
            continue
        global_idx_2 = globals_start_index_toadd + global_raw_idx_2

        edges_ls.append((global_idx_1, global_idx_2))
        edges_ls.append((global_idx_2, global_idx_1))
        counter = counter + 1
        if counter % 5000 == 0:
            logging.info("Inserted " + str(counter) + " "  + nyms_name + " edges")

    return edges_ls