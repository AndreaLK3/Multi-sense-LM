import pandas as pd
import Utils
import logging
import GetInputData.GetWordData as GWD
import GetInputData.BabelNet as BabelNet
import os


# Before starting: clean all storage files; reset vocabulary index to 0
def reset():

    input_and_processing_files = os.listdir(Utils.FOLDER_INPUT)
    all_files = [Utils.VOCAB_WT2_FILE, Utils.VOCAB_WT103_FILE] + input_and_processing_files

    for file in all_files:
        open(file, 'w')
    with open(Utils.VOCAB_CURRENT_INDEX_FILE, 'w') as vi_file:
        vi_file.write("0")


def continue_retrieving_data():
    Utils.init_logging(os.path.join("GetInputData", "RetrieveInputData.log"), logging.INFO)

    BN_request_sender = BabelNet.BabelNet_RequestSender()  # keeps track of the number of requests sent to BabelNet
    with open(Utils.VOCAB_CURRENT_INDEX_FILE, "r") as currentIndex_file:
        current_index = int(currentIndex_file.readline().strip())   # where were we?
    logging.info(current_index)

    # define and open (in 'append') the output archives for the KB data
    storage_filenames = [categ + ".h5" for categ in Utils.CATEGORIES]
    storage_filepaths = list(map(lambda fn: os.path.join(Utils.FOLDER_INPUT, fn), storage_filenames))
    open_storage_files = [pd.HDFStore(fname, mode='a') for fname in storage_filepaths]

    vocabulary_df = pd.read_hdf(Utils.VOCAB_WT2_FILE, mode='r')
    vocabulary_chunk = []


    while BN_request_sender.requests_counter < BN_request_sender.requests_threshold:

        word = vocabulary_df[vocabulary_df['index'] == current_index]
        logging.info("Retrieving Multisense data for word: " + str(word))
        logging.info("BN_request_sender.requests_counter= " + str(BN_request_sender.requests_counter))
        vocabulary_chunk.append(word)

        GWD.getAndSave_multisense_data(word, BN_request_sender, open_storage_files)


    # stop; we are approaching the maximum number of BabelNet requests (currently 5000)
    # save the index of the current word. We will proceed from there
    with open(Utils.VOCAB_CURRENT_INDEX_FILE, "r") as currentIndex_file:
        currentIndex_file.write(current_index)

    for storage_file in open_storage_files:
        storage_file.close()

    return vocabulary_chunk
