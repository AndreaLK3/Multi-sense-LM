import pandas as pd
import string
import Filesystem
import Lexicon
import Utils
import logging
import GetKBData.GetWordData as GWD
# import GetKBData.BabelNet as BabelNet
import os

def is_only_punctuation(word_token):
    if len(word_token) <=1:
        return True
    if all(list(map(lambda c: c in string.punctuation, word_token))):
        return True
    return False


def retrieve_data_WordNet(vocabulary_df, inputdata_folder, vocabulary_folder):

    with open(os.path.join(vocabulary_folder, Filesystem.VOCAB_CURRENT_INDEX_FILE), "r") as vi_file:
        current_index = int(vi_file.readline().strip())   # where were we?
    logging.info(current_index)

    requests_segment_size = 100000
    requests_counter = 0

    # define and open (in 'append') the output archives for the KB data
    storage_filenames = [categ + ".h5" for categ in Lexicon.CATEGORIES]
    storage_filepaths = list(map(lambda fn: os.path.join(inputdata_folder, fn), storage_filenames))
    open_storage_files = [pd.HDFStore(fname, mode='a') for fname in storage_filepaths]

    vocabulary_chunk = []

    while requests_counter < requests_segment_size:

        try:
            word = vocabulary_df.iloc[current_index]['word'] # causes exception when we finish reading the vocabulary
        except IndexError:
            logging.info("Finished retrieving input data for the " + str(requests_counter) +
                         " words of the vocabulary.")
            break
        logging.debug("RetrieveInputData.retrieve_data_WordNet() > " +
                     "word = vocabulary_df.iloc[current_index]['word']. >> Word="+str(word))

        requests_counter = requests_counter+1
        current_index = current_index + 1
        if is_only_punctuation(word):
            continue  # do not retrieve dictionary data for punctuation symbols, e.g. '==', '(' etc.
        if requests_counter % 100 == 0:
            logging.info("Retrieving WordNet senses data for word: " + str(word) + " - n." + str(requests_counter))

        vocabulary_chunk.append(word)
        # ********* core function invocation *********
        GWD.getAndSave_multisense_data(word, open_storage_files)
        # *********

    # save the index of the current word. We will proceed from there
    with open(os.path.join(vocabulary_folder, Filesystem.VOCAB_CURRENT_INDEX_FILE), "w") as currentIndex_file:
        currentIndex_file.write(str(current_index))

    for storage_file in open_storage_files:
        storage_file.close()

    return vocabulary_chunk