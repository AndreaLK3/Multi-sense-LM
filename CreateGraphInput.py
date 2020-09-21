import Filesystem as F
import GetKBInputData.RetrieveInputData as RID
import PrepareKBInput.PrepareKBInput as PI
import Utils
import os
import pandas as pd
import Vocabulary.Vocabulary as V
import WordEmbeddings.ComputeEmbeddings as CE
import tables
import logging
import SenseLabeledCorpus as SLC

# Before starting: clean all storage files; reset vocabulary index to 0
def reset():

    # reset the hdf5 archives for dictionary information: definitions, examples, synonyms, antonyms
    archives_filenames = list(filter(lambda fname: 'h5' in fname, os.listdir(F.FOLDER_INPUT)))
    archives_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_INPUT, fname), archives_filenames))

    # reset the modified training corpus, where we added the phrases
    phrased_corpus_filenames = [F.PHRASED_TRAINING_CORPUS, F.TEMPORARY_PHRASED_CORPUS]
    phrased_corpus_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_TEXT_CORPUSES, fname), phrased_corpus_filenames))

    # reset the vocabularies, and the SQL DB with the indices for the embedding matrices
    vocab_filepaths = list(map(lambda fname: os.path.join(F.FOLDER_VOCABULARY, fname),
                               [F.VOCABULARY_OF_GLOBALS_FILE, F.VOCAB_PHRASED]))
    db_filepaths = [os.path.join(F.FOLDER_INPUT, Utils.INDICES_TABLE_DB)]

    # reset the graph object file, and the area_matrices
    graph_filepaths = [os.path.join(F.FOLDER_NN, F.KBGRAPH_FILE)] + \
                      [os.path.join(F.FOLDER_GRAPH, fname) for fname in os.listdir(F.FOLDER_GRAPH) if '.npz' in fname]

    for fpath in archives_filepaths:
        f = pd.HDFStore(fpath, mode='w')
        f.close()

    for fpath in vocab_filepaths + phrased_corpus_filepaths + db_filepaths + graph_filepaths:
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



def exe_from_input_to_vectors(do_reset, compute_single_prototype, sp_method, vocabulary_from_senselabeled):
    Utils.init_logging('Pipeline_CGI.log')
    if do_reset:
        reset()

    vocab_text_fname = os.listdir(os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_MYTEXTCORPUS, F.FOLDER_TRAIN))[0]
    vocab_text_fpath = os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_MYTEXTCORPUS, F.FOLDER_TRAIN, vocab_text_fname)
    outvocab_filepath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE)
    if vocabulary_from_senselabeled: SLC.organize_splits()
    vocabulary = V.get_vocabulary_df(senselabeled_or_text=vocabulary_from_senselabeled,
                                     corpus_txt_fpaths=[vocab_text_fpath],
                                     out_vocabulary_h5_filepath=outvocab_filepath, min_count=2, lowercase=True)
    if compute_single_prototype:
        reset_embeddings()
        single_prototypes_file = F.SPVs_FASTTEXT_FILE if sp_method==CE.Method.FASTTEXT else F.SPVs_DISTILBERT_FILE
        CE.compute_single_prototype_embeddings(vocabulary,
                                               os.path.join(F.FOLDER_INPUT, single_prototypes_file),
                                               sp_method)

    vocabulary_ls = RID.retrieve_data_WordNet(vocabulary)
    logging.info("CreateGraphInput.exe() > "
                 + " number of ords included in the vocabulary chunk, to be prepared: " + str(len(vocabulary_ls)))
    PI.prepare(vocabulary_ls, sp_method)
    tables.file._open_files.close_all()