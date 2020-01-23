import Filesystem as F
import GetKBInputData.RetrieveInputData as RID
import PrepareKBInput.PrepareKBInput as PI
import Utils
import os
import pandas as pd
import Vocabulary.Vocabulary as V
import Vocabulary.Phrases as PHR
import WordEmbeddings.ComputeEmbeddings as CE
import tables
import logging

# Before starting: clean all storage files; reset vocabulary index to 0
def reset():

    # reset the hdf5 archives for dictionary information: definitions, examples, synonyms, antonyms
    archives_core_filenames = Utils.CATEGORIES + \
                    list(map(lambda c: Utils.PROCESSED + '_' + c, Utils.CATEGORIES))
    archives_filenames = list(map(lambda core_fname : core_fname + '.h5', archives_core_filenames))
    archives_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_INPUT, fname), archives_filenames))

    # reset the modified training corpus, where we added the phrases
    phrased_corpus_filenames = [F.PHRASED_TRAINING_CORPUS, F.TEMPORARY_PHRASED_CORPUS]
    phrased_corpus_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_TEXT_CORPUSES, fname), phrased_corpus_filenames))

    # reset the vocabularies, and the SQL DB with the indices for the embedding matrices
    vocab_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_VOCABULARY, fname),
                               [F.VOCABULARY_OF_GLOBALS_FILE, F.VOCAB_PHRASED]))
    db_filepaths = [os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)]

    for fpath in archives_filepaths:
        f = pd.HDFStore(fpath, mode='w')
        f.close()

    for fpath in vocab_filepaths + phrased_corpus_filepaths + db_filepaths:
        if os.path.exists(fpath):
            os.remove(fpath)

    with open(os.path.join(F.FOLDER_VOCABULARY, F.VOCAB_CURRENT_INDEX_FILE), 'w') as vi_file:
        vi_file.write("0")
        vi_file.close()


def reset_embeddings():
    # reset the embeddings, both those for dictionary elements and those for single-prototype vectors
    vectorized_inputs_filenames = list(filter(lambda fname: '.npy' in fname, os.listdir(F.FOLDER_INPUT)))
    vectorized_inputs_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_INPUT, fname),
                                           vectorized_inputs_filenames))
    for fpath in vectorized_inputs_filepaths:
        if os.path.exists(fpath):
            os.remove(fpath)



def exe(do_reset=False, compute_single_prototype=False, vocabulary_from_senselabeled=False):
    Utils.init_logging('Pipeline_CGI.log')
    if do_reset:
        reset()

    vocab_text_source = os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_WT2, F.WT_TRAIN_FILE)
    outvocab_filepath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE)
    vocabulary = V.get_vocabulary_df(senselabeled_or_text=vocabulary_from_senselabeled, slc_split_name='training',
                                     corpus_txt_filepath=vocab_text_source,
                                     out_vocabulary_h5_filepath=outvocab_filepath, min_count=10)

    if compute_single_prototype:
        reset_embeddings()

        CE.compute_single_prototype_embeddings(vocabulary,
                                               os.path.join(F.FOLDER_INPUT, F.SPVs_FASTTEXT_FILE),
                                               CE.Method.FASTTEXT)

    kb_data_chunk = RID.retrieve_data_WordNet(vocabulary)
    logging.info("CreateGraphInput.exe() > "
                 + " Words included in the vocabulary chunk, to be prepared: " + str(kb_data_chunk))
    PI.prepare(kb_data_chunk, CE.Method.FASTTEXT)
    tables.file._open_files.close_all()