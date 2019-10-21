import PrepareGraphInput.RemoveQuasiDuplicates as RQD
import PrepareGraphInput.LemmatizeNyms as LN
import PrepareGraphInput.SenseDenominations as SD
import PrepareGraphInput.EmbedWithBERT as EWB
import os
import Utils
import pandas as pd
import logging
import sqlite3

# Phase 1 - Preprocessing: eliminating quasi-duplicate definitions and examples, and lemmatizing synonyms & antonyms
def preprocess(vocabulary):
    Utils.init_logging(os.path.join('PrepareGraphInput','PreprocessInput.log'), logging.INFO)

    # categories= [d., e., s., a.]
    hdf5_input_filepaths = [os.path.join(Utils.FOLDER_INPUT, categ + ".h5") for categ in Utils.CATEGORIES]
    hdf5_output_filepaths = [os.path.join(Utils.FOLDER_INPUT, Utils.PROCESSED + '_' + categ + ".h5")
                             for categ in Utils.CATEGORIES]

    input_dbs = [pd.HDFStore(input_filepath, mode='r') for input_filepath in hdf5_input_filepaths]
    processed_dbs = [pd.HDFStore(output_filepath, mode='a') for output_filepath in hdf5_output_filepaths]

    for word in vocabulary:
        logging.info("Eliminating quasi-duplicate definitions for the senses of the word: " + word)
        RQD.eliminate_duplicates_in_word(word, Utils.DEFINITIONS, input_dbs[0], processed_dbs[0])
        logging.info("Eliminating quasi-duplicate examples for the senses of the word: " + word)
        RQD.eliminate_duplicates_in_word(word, Utils.EXAMPLES, input_dbs[1], processed_dbs[1])

        logging.info("Lemmatizing synonyms for the senses of the word: " + word)
        LN.lemmatize_nyms_in_word(word, Utils.SYNONYMS, input_dbs[2], processed_dbs[2])
        logging.info("Lemmatizing antonyms for the senses of the word: " + word)
        LN.lemmatize_nyms_in_word(word, Utils.ANTONYMS, input_dbs[3], processed_dbs[3])

    Utils.close_list_of_files(input_dbs + processed_dbs)


# Phase 2 - Selecting, sorting and naming (noun.1, verb.4, etc.) the senses of each word
def assign_sense_names(vocabulary):
    Utils.init_logging('AssignSenseNames.log', logging.INFO)

    hdf5_input_filepaths = [os.path.join(Utils.FOLDER_INPUT, Utils.PROCESSED + '_' + categ + ".h5")
                             for categ in Utils.CATEGORIES]
    hdf5_output_filepaths = [os.path.join(Utils.FOLDER_INPUT, Utils.DENOMINATED + '_' + categ + ".h5")
                            for categ in Utils.CATEGORIES]

    input_dbs = [pd.HDFStore(input_filepath, mode='r') for input_filepath in hdf5_input_filepaths]
    denominated_dbs = [pd.HDFStore(output_filepath, mode='a') for output_filepath in hdf5_output_filepaths]

    for word in vocabulary:
        logging.info("Selecting, sorting and naming the senses of the word: " + word)
        SD.assign_senses_to_word(word, input_dbs, denominated_dbs)

    Utils.close_list_of_files(input_dbs + denominated_dbs)


# Phase 3 - Considering the wordSenses in the vocabulary, located in the archive of denominated definitions,
# establish a correspondence with an integer index.
# Moreover, counting the number of defs and examples, define start&end indices for the matrix of word embeddings.
def create_senses_vocabulary_table(vocabulary_words_ls):
    Utils.init_logging('CreateSensesVocabularyTable.log', logging.INFO)

    defs_input_filepath = os.path.join(Utils.FOLDER_INPUT, Utils.DENOMINATED + '_' + Utils.DEFINITIONS + ".h5")
    examples_input_filepath = os.path.join(Utils.FOLDER_INPUT, Utils.DENOMINATED + '_' + Utils.EXAMPLES + ".h5")
    defs_input_db = pd.HDFStore(defs_input_filepath, mode='r')
    examples_input_db = pd.HDFStore(examples_input_filepath, mode='r')

    output_filepath = os.path.join(Utils.FOLDER_INPUT, Utils.INDICES_TABLE + ".sql")
    outdb_reset = open(output_filepath, 'w'); outdb_reset.close()
    out_vocabTable_db = sqlite3.connect(output_filepath)
    out_vocabTable_db_c = out_vocabTable_db.cursor()
    out_vocabTable_db_c.execute('''CREATE TABLE vocabulary_table (  word varchar(127),
                                                                    sense varchar(63),
                                                                    vocab_index int,
                                                                    start_defs int,
                                                                    end_defs int,
                                                                    start_examples int,
                                                                    end_examples ints
                                                )''')
    my_vocabulary_index = 0
    start_defs_count = 0
    start_examples_count = 0

    for word in vocabulary_words_ls:
        word_defs_df = defs_input_db.select(key=Utils.DEFINITIONS, where="word == " + str(word))
        word_examples_df = examples_input_db.select(key=Utils.EXAMPLES, where="word == " + str(word))

        sense_names = set(word_defs_df['sense'])
        for sense in sense_names:
            sense_defs_df = word_defs_df.loc[word_defs_df['sense'] == sense]
            sense_examples_df = word_examples_df.loc[word_examples_df['sense'] == sense]

            end_defs_count = start_defs_count + len(sense_defs_df.index)
            end_examples_count = start_examples_count + len(sense_examples_df.index)
            out_vocabTable_db_c.execute("INSERT INTO vocabulary_table VALUES (?,?,?,?,?,?,?)", (word, sense, my_vocabulary_index,
                                                                                start_defs_count, end_defs_count,
                                                                                start_examples_count, end_examples_count))

            # update counters
            my_vocabulary_index = my_vocabulary_index + 1
            start_defs_count = end_defs_count
            start_examples_count = end_examples_count

        out_vocabTable_db.commit()
    out_vocabTable_db.close()


# ['move', 'light']
def prepare(vocabulary):
    Utils.init_logging(os.path.join("PrepareGraphInput", "PrepareInput"))

    hdf5_output_filepaths = [os.path.join(Utils.FOLDER_INPUT, Utils.PROCESSED + '_' + categ + ".h5")
                             for categ in Utils.CATEGORIES] + \
                            [os.path.join(Utils.FOLDER_INPUT, Utils.DENOMINATED + '_' + categ + ".h5")
                             for categ in Utils.CATEGORIES] +\
                            [os.path.join(Utils.FOLDER_INPUT, Utils.VECTORIZED + '_' + categ + ".npy")
                             for categ in Utils.CATEGORIES[0:2]]

    all_archives = [open(input_filepath, 'w') for input_filepath in hdf5_output_filepaths] # we must reset the output archivess
    Utils.close_list_of_files(all_archives)

    # Phase 1 - Preprocessing: eliminating quasi-duplicate definitions and examples, and lemmatizing synonyms & antonyms
    preprocess(vocabulary)

    # Phase 2 - Selecting, sorting and naming (noun.1, verb.4, etc.) the senses of each word
    assign_sense_names(vocabulary)

    # Phase 3 - Create the Vocabulary table with the correspondences (wordSense, integer index).
    create_senses_vocabulary_table(vocabulary)

    # Phase 4 - get the sentence embeddings for definitions and examples, using BERT, and store them
    EWB.compute_sentence_embeddings(Utils.DEFINITIONS)
    EWB.compute_sentence_embeddings(Utils.EXAMPLES)

    Utils.close_list_of_files(all_archives)
