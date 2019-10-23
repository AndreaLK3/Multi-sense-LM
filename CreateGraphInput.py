import Filesystem as F
import GetInputData.RetrieveInputData as RID
import PrepareGraphInput.PrepareInput as PI
import Utils
import os
import pandas as pd
import Vocabulary.Vocabulary as VOC
import Vocabulary.Phrases as PHR


# Before starting: clean all storage files; reset vocabulary index to 0
def reset():
    input_corenames = Utils.CATEGORIES + list(map(lambda c: Utils.DENOMINATED + '_' + c, Utils.CATEGORIES)) + \
                    list(map(lambda c: Utils.PROCESSED + '_' + c, Utils.CATEGORIES)) + \
                      [F.BN_WORD_INTROS, F.BN_SYNSET_DATA, F.BN_SYNSET_EDGES]
    input_filenames = list(map(lambda corename : corename + '.h5', input_corenames)) + [F.PHRASED_TRAINING_CORPUS]
    input_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_INPUT, fname),
                                              input_filenames))
    vocab_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_VOCABULARY, fname),
                               [F.VOCAB_WT2_FILE, F.VOCAB_WT103_FILE]))

    for fpath in input_filepaths:
        if fpath.endswith('h5'):
            f = pd.HDFStore(fpath, mode='w')
            f.close()
        else:
            if os.path.exists(fpath):
                os.remove(fpath)
    for fpath in vocab_filepaths:
        if os.path.exists(fpath):
            os.remove(fpath)
    with open(os.path.join(F.FOLDER_VOCABULARY, F.VOCAB_CURRENT_INDEX_FILE), 'w') as vi_file:
        vi_file.write("0")
        vi_file.close()


def exe(do_reset=False):
    Utils.init_logging('CreateGraphInput.log')
    if do_reset:
        reset()
    PHR.setup_phrased_corpus(os.path.join(F.FOLDER_WT2, F.WT_TRAIN_FILE),
                             os.path.join(F.FOLDER_INPUT, F.PHRASED_TRAINING_CORPUS))
    VOC.get_vocabulary_df(os.path.join(F.FOLDER_VOCABULARY, F.VOCAB_WT2_FILE),
                          os.path.join(F.FOLDER_INPUT, F.PHRASED_TRAINING_CORPUS), min_count=5)

    vocabulary_chunk = RID.continue_retrieving_data()
    PI.prepare(vocabulary_chunk)