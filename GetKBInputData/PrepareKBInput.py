import GetKBInputData.RemoveQuasiDuplicates as RQD
import GetKBInputData.LemmatizeNyms as LN
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import os
import Utils
import pandas as pd
import logging
import sqlite3
import Filesystem as F
import numpy as np
from sklearn.decomposition import PCA
import nltk

# Phase 1 - Preprocessing: eliminating quasi-duplicate definitions and examples, and lemmatizing synonyms & antonyms
def preprocess(vocabulary_ls, inputdata_folder):

    # categories= [d., e., s., a.]
    hdf5_input_filepaths = [os.path.join(inputdata_folder, categ + ".h5") for categ in Utils.CATEGORIES]
    hdf5_output_filepaths = [os.path.join(inputdata_folder, Utils.PROCESSED + '_' + categ + ".h5")
                             for categ in Utils.CATEGORIES]

    input_dbs = [pd.HDFStore(input_filepath, mode='r') for input_filepath in hdf5_input_filepaths]
    processed_dbs = [pd.HDFStore(output_filepath, mode='a') for output_filepath in hdf5_output_filepaths]

    logging.info("Eliminating quasi-duplicate definitions for the current vocabulary subset.")
    RQD.eliminate_duplicates_in_table(vocabulary_ls, Utils.DEFINITIONS, input_dbs[0], processed_dbs[0])
    logging.info("Eliminating quasi-duplicate examples for the current vocabulary subset.")
    RQD.eliminate_duplicates_in_table(vocabulary_ls, Utils.EXAMPLES, input_dbs[1], processed_dbs[1])
    logging.info("Lemmatizing synonyms for the current vocabulary subset.")
    LN.lemmatize_nyms_in_word(vocabulary_ls, Utils.SYNONYMS, input_dbs[2], processed_dbs[2])
    logging.info("Lemmatizing antonyms for the current vocabulary subset.")
    LN.lemmatize_nyms_in_word(vocabulary_ls, Utils.ANTONYMS, input_dbs[3], processed_dbs[3])

    Utils.close_list_of_files(input_dbs + processed_dbs)


# Phase 2 - Considering the wordSenses in the vocabulary, located in the archive of processed definitions,
# establish a correspondence with an integer index.
# Moreover, counting the number of defs and examples, define start&end indices for the matrix of word embeddings.
def create_senses_indices_table(input_folder_fpath, vocabulary_folder):

    # ------- Setting filepaths --------
    vocab_fpath = os.path.join(vocabulary_folder, "vocabulary.h5")
    vocabulary_df = pd.read_hdf(vocab_fpath)
    vocabulary_words_ls = vocabulary_df['word'].to_list().copy()
    vocabulary_lemmatizedforms_ls = vocabulary_df['lemmatized_form'].to_list().copy()

    defs_input_filepath = os.path.join(input_folder_fpath, Utils.PROCESSED + '_' + Utils.DEFINITIONS + ".h5")
    examples_input_filepath = os.path.join(input_folder_fpath, Utils.PROCESSED + '_' + Utils.EXAMPLES + ".h5")
    defs_input_db = pd.HDFStore(defs_input_filepath, mode='r')
    examples_input_db = pd.HDFStore(examples_input_filepath, mode='r')

    # ------- Creating the table --------
    output_filepath = os.path.join(input_folder_fpath, Utils.INDICES_TABLE_DB)
    out_indicesTable_db = sqlite3.connect(output_filepath)
    out_indicesTable_db_c = out_indicesTable_db.cursor()
    out_indicesTable_db_c.execute('''CREATE TABLE IF NOT EXISTS
                                                indices_table (  word_sense varchar(127),
                                                                    vocab_index int,
                                                                    start_defs int,
                                                                    end_defs int,
                                                                    start_examples int,
                                                                    end_examples ints
                                                )''')
    my_vocabulary_index = 0
    start_defs_count = 0
    start_examples_count = 0

    logging.debug("vocabulary_words_ls=" + str(vocabulary_words_ls))

    # ------- Inserting the words from the vocabulary that have sense/WordNet data, in the standard way --------
    word_senses_series_from_defs = defs_input_db[Utils.DEFINITIONS][Utils.SENSE_WN_ID]
    word_senses_ls = [sense_str for sense_str in word_senses_series_from_defs if
                             Utils.get_word_from_sense(sense_str) in vocabulary_words_ls]
    words_with_senses_set = set()
    for wn_id in word_senses_ls:
        logging.debug('GetKBInputData.create_senses_vocabulary_table(vocabulary_words_ls) > word_senses_toprocess > '
                     + ' current wn_id=' + wn_id)
        sense_defs_df = Utils.select_from_hdf5(defs_input_db, Utils.DEFINITIONS, [Utils.SENSE_WN_ID], [wn_id])
        sense_examples_df = Utils.select_from_hdf5(examples_input_db, Utils.EXAMPLES, [Utils.SENSE_WN_ID], [wn_id])


        end_defs_count = start_defs_count + len(sense_defs_df.index)
        end_examples_count = start_examples_count + len(sense_examples_df.index)
        out_indicesTable_db_c.execute("INSERT INTO indices_table VALUES (?,?,?,?,?,?)", (wn_id, my_vocabulary_index,
                                                                            start_defs_count, end_defs_count,
                                                                            start_examples_count, end_examples_count))

        # update counters
        my_vocabulary_index = my_vocabulary_index + 1
        start_defs_count = end_defs_count
        start_examples_count = end_examples_count
        # add the word to the set of words that do have a sense
        words_with_senses_set.add(Utils.get_word_from_sense(wn_id))

    words_without_senses_set = set(vocabulary_words_ls).difference(words_with_senses_set)
    logging.info("words_without_senses_set=" + str(words_without_senses_set))

    # ------- Remove w from the words needing a dummySense if the lemmatized form of w has senses  --------
    words_without_senses_ls = list(words_without_senses_set)
    words_needing_dummySense_ls = []
    for word in words_without_senses_ls:
        word_index = vocabulary_words_ls.index(word)
        lemmatized_form = vocabulary_lemmatizedforms_ls[word_index]
        if lemmatized_form in words_with_senses_set:
            logging.info("'"+word + "' does not need a dummySense, because '" + lemmatized_form + "' is its lemmatized parent"
                         + " and has senses already.")
            continue
        else:
            logging.info(word + " needs a dummySense, the lemmatized form '" + str(lemmatized_form)+"' has none")
            words_needing_dummySense_ls.append(word)
    words_needing_dummySense_set = set(words_needing_dummySense_ls)

    logging.info("words_needing_dummySense_set=" + str(words_needing_dummySense_set))

    # ------- Creating and inserting the dummySenses --------
    for word in words_needing_dummySense_set:
        # no definitions nor examples to add here. We will add the global vector as the vector of the dummy-sense.
        dummy_wn_id = word + '.' + 'dummySense' + '.01'
        logging.debug("dummy_wn_id=" + str(dummy_wn_id))
        end_defs_count = start_defs_count
        end_examples_count = start_examples_count
        out_indicesTable_db_c.execute("INSERT INTO indices_table VALUES (?,?,?,?,?,?)", (dummy_wn_id, my_vocabulary_index,
                                                                                         start_defs_count,
                                                                                         end_defs_count,
                                                                                         start_examples_count,
                                                                                         end_examples_count))

        logging.debug("VocabularyAndEmbeddings index of the sense " + dummy_wn_id + " = " + str(my_vocabulary_index))
        logging.debug("start_defs_count=" + str(start_defs_count) + " ; end_defs_count=" + str(end_defs_count) +
                      " ; start_examples_count=" + str(start_examples_count) + " ; end_examples_count=" + str(
            end_examples_count))
        my_vocabulary_index = my_vocabulary_index + 1

    out_indicesTable_db.commit()
    out_indicesTable_db.close()

# Phase 4 - PCA dimensionality reduction, for definitions and examples. Writing to files in subfolder.
# PCA: Linear dimensionality reduction, using Singular Value Decomposition of the data to project
# it to a lower dimensional space. The input data is centered but not scaled for each feature before applying SVD.
def apply_PCA_to_defs_examples(embeddings_method, inputdata_folder):
    elements_ls = [Utils.DEFINITIONS, Utils.EXAMPLES]
    for elements_name in elements_ls:
        elems_fname = Utils.VECTORIZED + '_' + str(embeddings_method.value) + '_' + elements_name + '.npy'
        elems_fpath = os.path.join(inputdata_folder, elems_fname)
        elems_E1 = np.load(elems_fpath)

        pca = PCA(n_components=Utils.GRAPH_EMBEDDINGS_DIM)
        elems_E2 = pca.fit_transform(elems_E1)

        output_dir = Utils.set_directory(os.path.join(inputdata_folder, F.FOLDER_PCA))
        output_fpath = os.path.join(output_dir, elements_name + '_' + str(embeddings_method.value) + '.npy')
        np.save(output_fpath, elems_E2)
    return


def prepare(vocabulary_ls, inputdata_folder, vocabulary_folder, embeddings_method):

    # Phase 1 - Preprocessing: eliminating quasi-duplicate definitions and examples, and lemmatizing synonyms & antonyms
    preprocess(vocabulary_ls, inputdata_folder)

    # Phase 2 - Create the VocabularyAndEmbeddings table with the correspondences (wordSense, integer index).
    create_senses_indices_table(inputdata_folder, vocabulary_folder)

    # Phase 3 - get the sentence embeddings for definitions and examples, using BERT or FasText, and store them
    CE.compute_elements_embeddings(Utils.DEFINITIONS, embeddings_method, inputdata_folder)
    CE.compute_elements_embeddings(Utils.EXAMPLES, embeddings_method, inputdata_folder)

    # Phase 4 - use PCA dimensionality reduction for definitions and examples, for future graph processing
    apply_PCA_to_defs_examples(embeddings_method, inputdata_folder)