import pandas as pd
import Utils
import logging
import GetInputData.GetWordData as GWD
import GetInputData.BabelNet as BabelNet
import os


def continue_retrieving_data():
    #Utils.init_logging(os.path.join("GetInputData", "RetrieveInputData.log"), logging.INFO)

    BN_request_sender = BabelNet.BabelNet_RequestSender()  # keeps track of the number of requests sent to BabelNet
    with open(os.path.join(Utils.FOLDER_VOCABULARY,Utils.VOCAB_CURRENT_INDEX_FILE), "r") as vi_file:
        current_index = int(vi_file.readline().strip())   # where were we?
    logging.info(current_index)

    # define and open (in 'append') the output archives for the KB data
    storage_filenames = [categ + ".h5" for categ in Utils.CATEGORIES]
    storage_filepaths = list(map(lambda fn: os.path.join(Utils.FOLDER_INPUT, fn), storage_filenames))
    open_storage_files = [pd.HDFStore(fname, mode='a') for fname in storage_filepaths]

    vocabulary_df = pd.read_hdf(os.path.join(Utils.FOLDER_VOCABULARY,Utils.VOCAB_WT2_FILE), mode='r')
    vocabulary_chunk = []


    while BN_request_sender.requests_counter < BN_request_sender.requests_threshold:

        word = vocabulary_df.iloc[current_index]['word']
        logging.info("Retrieving Multisense data for word: " + str(word))
        logging.info("BN_request_sender.requests_counter= " + str(BN_request_sender.requests_counter))
        vocabulary_chunk.append(word)

        GWD.getAndSave_multisense_data(word, BN_request_sender, open_storage_files)
        current_index = current_index + 1

    logging.info("BN_request_sender.requests_counter= " + str(BN_request_sender.requests_counter))
    # stop; we are approaching the maximum number of BabelNet requests (currently 5000)
    # save the index of the current word. We will proceed from there
    with open(os.path.join(Utils.FOLDER_VOCABULARY,Utils.VOCAB_CURRENT_INDEX_FILE), "w") as currentIndex_file:
        currentIndex_file.write(str(current_index))

    for storage_file in open_storage_files:
        storage_file.close()

    return vocabulary_chunk
