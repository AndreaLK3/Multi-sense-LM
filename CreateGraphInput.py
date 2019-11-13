import Filesystem as F
import GetInputData.RetrieveInputData as RID
import PrepareGraphInput.PrepareInput as PI
import Utils
import os
import pandas as pd
import Vocabulary.Vocabulary as VOC
import Vocabulary.Phrases as PHR
import WordEmbeddings.SinglePrototypeVectors as SPV

# Before starting: clean all storage files; reset vocabulary index to 0
def reset():

    # reset: hdf5 archives for dictionary information: definitions, examples, synonyms, antonyms
    archives_core_filenames = Utils.CATEGORIES + list(map(lambda c: Utils.DENOMINATED + '_' + c, Utils.CATEGORIES)) + \
                    list(map(lambda c: Utils.PROCESSED + '_' + c, Utils.CATEGORIES))
    archives_filenames = list(map(lambda core_fname : core_fname + '.h5', archives_core_filenames))
    archives_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_INPUT, fname), archives_filenames))

    # reset: modified training corpus, where we added the phrases
    phrased_corpus_filenames = [F.PHRASED_TRAINING_CORPUS, F.TEMPORARY_PHRASED_CORPUS]
    phrased_corpus_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_TEXT_CORPUSES, fname), phrased_corpus_filenames))

    vectorized_inputs_filenames = list(filter(lambda fname: '.npy' in fname, os.listdir(F.FOLDER_INPUT)))
    vectorized_inputs_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_INPUT, fname),
                                              vectorized_inputs_filenames))
    vocab_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_VOCABULARY, fname),
                               [F.VOCAB_WT2_FILE, F.VOCAB_WT103_FILE, F.VOCAB_PHRASED]))

    for fpath in archives_filepaths:
        f = pd.HDFStore(fpath, mode='w')
        f.close()

    for fpath in vectorized_inputs_filepaths + vocab_filepaths + phrased_corpus_filepaths:
        if os.path.exists(fpath):
            os.remove(fpath)

    with open(os.path.join(F.FOLDER_VOCABULARY, F.VOCAB_CURRENT_INDEX_FILE), 'w') as vi_file:
        vi_file.write("0")
        vi_file.close()


def exe(do_reset=False):
    Utils.init_logging('CreateGraphInput.log')
    if do_reset:
        reset()

    # NOTE: For the sake of 1) Building a Version 1.0 ; 2) Simplicity and streamlining ; -> we ignore Phrases for now
    # PHR.setup_phrased_corpus(os.path.join(F.FOLDER_WT2, F.WT_TRAIN_FILE),
    #                          os.path.join(F.FOLDER_INPUT, F.PHRASED_TRAINING_CORPUS),
    #                          min_freq=40, score_threshold=120) # bigrams of phrases
    # VOC.get_vocabulary_df(os.path.join(F.FOLDER_VOCABULARY, F.VOCAB_PHRASED),
    #                       os.path.join(F.FOLDER_TEXT_CORPUSES, F.PHRASED_TRAINING_CORPUS), min_count=5)

    VOC.get_vocabulary_df(os.path.join(F.FOLDER_VOCABULARY, F.VOCAB_WT2_FILE),
                          os.path.join(F.FOLDER_WT2, F.WT_TRAIN_FILE), min_count=5)

    SPV.compute_single_prototype_embeddings(os.path.join(F.FOLDER_VOCABULARY, F.VOCAB_WT2_FILE))

    kb_data_chunk = RID.continue_retrieving_data()
    PI.prepare(kb_data_chunk)