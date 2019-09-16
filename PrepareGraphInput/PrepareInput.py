import PrepareGraphInput.RemoveQuasiDuplicates as RQD
import PrepareGraphInput.LemmatizeNyms as LN
import PrepareGraphInput.SenseDenominations as SD
import os
import Utils
import pandas as pd
import logging

def preprocess(vocabulary):
    Utils.init_logging('PreprocessInput.log', logging.INFO)

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



def assign_sense_names(vocabulary = ['wide', 'plant', 'move', 'light']):
    Utils.init_logging('AssignSenseNames.log', logging.INFO)

    hdf5_input_filepaths = [os.path.join(Utils.FOLDER_INPUT, Utils.PROCESSED + '_' + categ + ".h5")
                             for categ in Utils.CATEGORIES]
    hdf5_output_filepaths = [os.path.join(Utils.FOLDER_INPUT, Utils.DENOMINATED + '_' + categ + ".h5")
                            for categ in Utils.CATEGORIES]

    input_dbs = [pd.HDFStore(input_filepath, mode='r') for input_filepath in hdf5_input_filepaths]
    denominated_dbs = [pd.HDFStore(output_filepath, mode='a') for output_filepath in hdf5_output_filepaths]


    for word in vocabulary:
        SD.assign_senses_to_word(word, input_dbs, denominated_dbs)



