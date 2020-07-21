import Filesystem
import PrepareKBInput.RemoveQuasiDuplicates as RQD
import PrepareKBInput.LemmatizeNyms as LN
import PrepareKBInput.SenseDenominations as SD
import WordEmbeddings.ComputeEmbeddings as CE
import WordEmbeddings.EmbedWithDBERT as EWB
import os
import Utils
import pandas as pd
import logging
import sqlite3
import re

# Phase 1 - Preprocessing: eliminating quasi-duplicate definitions and examples, and lemmatizing synonyms & antonyms
def preprocess(vocabulary_ls):
    #Utils.init_logging(os.path.join('PrepareKBInput','PreprocessInput.log'), logging.INFO)

    # categories= [d., e., s., a.]
    hdf5_input_filepaths = [os.path.join(Filesystem.FOLDER_INPUT, categ + ".h5") for categ in Utils.CATEGORIES]
    hdf5_output_filepaths = [os.path.join(Filesystem.FOLDER_INPUT, Utils.PROCESSED + '_' + categ + ".h5")
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
def create_senses_indices_table(vocabulary_words_ls):
    #Utils.init_logging('CreateSensesVocabularyTable.log', logging.INFO)

    defs_input_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.PROCESSED + '_' + Utils.DEFINITIONS + ".h5")
    examples_input_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.PROCESSED + '_' + Utils.EXAMPLES + ".h5")
    defs_input_db = pd.HDFStore(defs_input_filepath, mode='r')
    examples_input_db = pd.HDFStore(examples_input_filepath, mode='r')

    output_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
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

    # Process the wn_ids as previously > words_without_sense_set > add to the table with type=n and only vocab_index+1
    word_senses_series_from_defs = defs_input_db[Utils.DEFINITIONS][Utils.SENSE_WN_ID]
    word_senses_ls = [sense_str for sense_str in word_senses_series_from_defs if
                             Utils.get_word_from_sense(sense_str) in vocabulary_words_ls]
    words_with_senses_set = set()
    for wn_id in word_senses_ls:
        logging.debug('PrepareKBInput.create_senses_vocabulary_table(vocabulary_words_ls) > word_senses_toprocess > '
                     + ' current wn_id=' + wn_id)
        sense_defs_df = Utils.select_from_hdf5(defs_input_db, Utils.DEFINITIONS, [Utils.SENSE_WN_ID], [wn_id])
        sense_examples_df = Utils.select_from_hdf5(examples_input_db, Utils.EXAMPLES, [Utils.SENSE_WN_ID], [wn_id])


        end_defs_count = start_defs_count + len(sense_defs_df.index)
        end_examples_count = start_examples_count + len(sense_examples_df.index)
        out_indicesTable_db_c.execute("INSERT INTO indices_table VALUES (?,?,?,?,?,?)", (wn_id, my_vocabulary_index,
                                                                            start_defs_count, end_defs_count,
                                                                            start_examples_count, end_examples_count))

        logging.debug("Vocabulary index of the sense " + wn_id + " = " + str(my_vocabulary_index))
        logging.debug("start_defs_count=" + str(start_defs_count) + " ; end_defs_count=" + str(end_defs_count) +
                     " ; start_examples_count=" + str(start_examples_count) + " ; end_examples_count=" + str(end_examples_count))

        # update counters
        my_vocabulary_index = my_vocabulary_index + 1
        start_defs_count = end_defs_count
        start_examples_count = end_examples_count
        # add the word to the set of words that do have a sense
        words_with_senses_set.add(wn_id[0:wn_id.find('.')])

    words_without_senses_set = set(vocabulary_words_ls).difference(words_with_senses_set)

    for word in words_without_senses_set:
        # no definitions nor examples to add here. We will add the global vector as the vector of the dummy-sense.
        dummy_wn_id = word + '.' + 'dummySense' + '.01'
        end_defs_count = start_defs_count
        end_examples_count = start_examples_count
        out_indicesTable_db_c.execute("INSERT INTO indices_table VALUES (?,?,?,?,?,?)", (dummy_wn_id, my_vocabulary_index,
                                                                                         start_defs_count,
                                                                                         end_defs_count,
                                                                                         start_examples_count,
                                                                                         end_examples_count))

        logging.debug("Vocabulary index of the sense " + dummy_wn_id + " = " + str(my_vocabulary_index))
        logging.debug("start_defs_count=" + str(start_defs_count) + " ; end_defs_count=" + str(end_defs_count) +
                      " ; start_examples_count=" + str(start_examples_count) + " ; end_examples_count=" + str(
            end_examples_count))
        my_vocabulary_index = my_vocabulary_index + 1

    out_indicesTable_db.commit()
    out_indicesTable_db.close()


# ['move', 'light']
def prepare(vocabulary_ls, embeddings_method): #vocabulary = ['move', 'light', 'for', 'sea']
    #Utils.init_logging(os.path.join("PrepareKBInput", "PrepareKBInput.log"))

    # Phase 1 - Preprocessing: eliminating quasi-duplicate definitions and examples, and lemmatizing synonyms & antonyms
    preprocess(vocabulary_ls)

    # Phase 2 - Create the Vocabulary table with the correspondences (wordSense, integer index).
    create_senses_indices_table(vocabulary_ls)

    # Phase 3 - get the sentence embeddings for definitions and examples, using BERT or FasText, and store them
    CE.compute_elements_embeddings(Utils.DEFINITIONS, embeddings_method)
    CE.compute_elements_embeddings(Utils.EXAMPLES, embeddings_method)
