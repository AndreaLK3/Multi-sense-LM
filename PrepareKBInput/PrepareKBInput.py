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
def create_senses_vocabulary_table(vocabulary_words_ls):
    #Utils.init_logging('CreateSensesVocabularyTable.log', logging.INFO)

    defs_input_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.PROCESSED + '_' + Utils.DEFINITIONS + ".h5")
    examples_input_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.PROCESSED + '_' + Utils.EXAMPLES + ".h5")
    defs_input_db = pd.HDFStore(defs_input_filepath, mode='r')
    examples_input_db = pd.HDFStore(examples_input_filepath, mode='r')

    output_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.INDICES_TABLE_DB)
    out_vocabTable_db = sqlite3.connect(output_filepath)
    out_vocabTable_db_c = out_vocabTable_db.cursor()
    out_vocabTable_db_c.execute('''CREATE TABLE IF NOT EXISTS
                                                vocabulary_table (  word_sense varchar(127),
                                                                    vocab_index int,
                                                                    start_defs int,
                                                                    end_defs int,
                                                                    start_examples int,
                                                                    end_examples ints
                                                )''')
    my_vocabulary_index = 0
    start_defs_count = 0
    start_examples_count = 0

    all_word_senses = defs_input_db[Utils.DEFINITIONS][Utils.SENSE_WN_ID]
    word_senses_toprocess = [sense_str for sense_str in all_word_senses if
                             Utils.get_word_from_sense(sense_str) in vocabulary_words_ls]
    for wn_id in word_senses_toprocess:
        logging.debug('PrepareKBInput.create_senses_vocabulary_table(vocabulary_words_ls) > word_senses_toprocess > '
                     + ' current wn_id=' + wn_id)
        sense_defs_df = Utils.select_from_hdf5(defs_input_db, Utils.DEFINITIONS, [Utils.SENSE_WN_ID], [wn_id])
        sense_examples_df = Utils.select_from_hdf5(examples_input_db, Utils.EXAMPLES, [Utils.SENSE_WN_ID], [wn_id])

        end_defs_count = start_defs_count + len(sense_defs_df.index)
        end_examples_count = start_examples_count + len(sense_examples_df.index)
        out_vocabTable_db_c.execute("INSERT INTO vocabulary_table VALUES (?,?,?,?,?,?)", (wn_id, my_vocabulary_index,
                                                                            start_defs_count, end_defs_count,
                                                                            start_examples_count, end_examples_count))

        logging.info("Vocabulary index of the sense " + wn_id + " = " + str(my_vocabulary_index))
        logging.info("start_defs_count=" + str(start_defs_count) + " ; end_defs_count=" + str(end_defs_count) +
                     " ; start_examples_count=" + str(start_examples_count) + " ; end_examples_count=" + str(end_examples_count))

        # update counters
        my_vocabulary_index = my_vocabulary_index + 1
        start_defs_count = end_defs_count
        start_examples_count = end_examples_count

    out_vocabTable_db.commit()
    out_vocabTable_db.close()


# ['move', 'light']
def prepare(vocabulary): #vocabulary = ['move', 'light', 'for', 'sea']
    #Utils.init_logging(os.path.join("PrepareKBInput", "PrepareKBInput.log"))

    # Phase 1 - Preprocessing: eliminating quasi-duplicate definitions and examples, and lemmatizing synonyms & antonyms
    preprocess(vocabulary)

    # Phase 2 - Create the Vocabulary table with the correspondences (wordSense, integer index).
    create_senses_vocabulary_table(vocabulary)

    # Phase 3a - get the sentence embeddings for definitions and examples, using BERT, and store them
    CE.compute_elements_embeddings(Utils.DEFINITIONS, CE.Method.DISTILBERT)
    CE.compute_elements_embeddings(Utils.EXAMPLES, CE.Method.DISTILBERT)

    # Phase 3b - get the sentence embeddings for definitions and examples, using FastText, and store them
    CE.compute_elements_embeddings(Utils.DEFINITIONS, CE.Method.FASTTEXT)
    CE.compute_elements_embeddings(Utils.EXAMPLES, CE.Method.FASTTEXT)