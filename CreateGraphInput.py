import GetInputData.RetrieveInputData as RID
import PrepareGraphInput.PrepareInput as PI
import Utils
import os
import pandas as pd
import Vocabulary.Vocabulary as VOC
import Vocabulary.Phrases as PHR

# Before starting: clean all storage files; reset vocabulary index to 0
def reset():

    input_and_processing_filepaths = list(map(lambda fname: os.path.join(Utils.FOLDER_INPUT, fname),
                                              os.listdir(Utils.FOLDER_INPUT)))
    vocab_filepaths = list(map(lambda fname: os.path.join(Utils.FOLDER_VOCABULARY, fname),
                               [Utils.VOCAB_WT2_FILE, Utils.VOCAB_WT103_FILE]))

    for fpath in input_and_processing_filepaths:
        if fpath.endswith('h5'):
            f = pd.HDFStore(fpath, mode='w')
            f.close()
        else:
            if os.path.exists(fpath):
                os.remove(fpath)
    for fpath in vocab_filepaths:
        if os.path.exists(fpath):
            os.remove(fpath)
    with open(os.path.join(Utils.FOLDER_VOCABULARY,Utils.VOCAB_CURRENT_INDEX_FILE), 'w') as vi_file:
        vi_file.write("0")
        vi_file.close()


def exe(do_reset=False):
    Utils.init_logging('CreateGraphInput.log')
    if do_reset:
        reset()
    PHR.setup_phrased_corpus(os.path.join(Utils.FOLDER_WT2, Utils.WT_TRAIN_FILE),
                             os.path.join(Utils.FOLDER_INPUT, Utils.PHRASED_TRAINING_CORPUS))
    VOC.get_vocabulary_df(os.path.join(Utils.FOLDER_VOCABULARY, Utils.VOCAB_WT2_FILE),
                        os.path.join(Utils.FOLDER_WT2, Utils.WT_TRAIN_FILE), min_count=2)
    vocabulary_chunk = RID.continue_retrieving_data()
    PI.prepare(vocabulary_chunk)