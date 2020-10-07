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
def reset(senselabeled_or_text):

    if senselabeled_or_text:
        input_folder = os.path.join(F.FOLDER_INPUT, F.FOLDER_SENSELABELED)
        vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, F.FOLDER_SENSELABELED)
        graph_folder = os.path.join(F.FOLDER_GRAPH, F.FOLDER_SENSELABELED)
    else:
        input_folder = os.path.join(F.FOLDER_INPUT, F.FOLDER_STANDARDTEXT)
        vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, F.FOLDER_STANDARDTEXT)
        graph_folder = os.path.join(F.FOLDER_GRAPH, F.FOLDER_STANDARDTEXT)

    # reset the hdf5 archives for dictionary information: definitions, examples, synonyms, antonyms
    archives_filenames = list(filter(lambda fname: 'h5' in fname, os.listdir(input_folder)))
    archives_filepaths = list(map(lambda fname: os.path.join(input_folder, fname), archives_filenames))
    for fpath in archives_filepaths:
        f = pd.HDFStore(fpath, mode='w')
        f.close()

    # reset the vocabularies, and the SQL DB with the indices for the embedding matrices
    vocab_filepaths = list(map(lambda fname: os.path.join(vocabulary_folder, fname),
                               [F.VOCABULARY_OF_GLOBALS_FILENAME]))
    db_filepaths = [os.path.join(input_folder, Utils.INDICES_TABLE_DB)]

    # reset the graph object file, and the area_matrices
    graph_filepaths = [os.path.join(graph_folder, F.KBGRAPH_FILE)] + \
                      [os.path.join(graph_folder, fname) for fname in os.listdir(F.FOLDER_GRAPH) if '.npz' in fname]

    for fpath in vocab_filepaths + db_filepaths + graph_filepaths:
        if os.path.exists(fpath):
            os.remove(fpath)

    with open(os.path.join(vocabulary_folder, F.VOCAB_CURRENT_INDEX_FILE), 'w') as vi_file:
        vi_file.write("0")
        vi_file.close()


def reset_embeddings(senselabeled_or_text):
    input_folder = os.path.join(F.FOLDER_INPUT, F.FOLDER_SENSELABELED) if senselabeled_or_text \
              else os.path.join(F.FOLDER_INPUT, F.FOLDER_STANDARDTEXT)
    # reset the embeddings, both those for dictionary elements and those for single-prototype vectors
    vectorized_inputs_filenames = list(filter(lambda fname: '.npy' in fname, os.listdir(input_folder)))
    vectorized_inputs_filepaths = list(map(lambda fname: os.path.join(input_folder, fname),
                                           vectorized_inputs_filenames))
    for fpath in vectorized_inputs_filepaths:
        if os.path.exists(fpath):
            os.remove(fpath)



def exe_from_input_to_vectors(do_reset, compute_single_prototype, senselabeled_or_text, sp_method=CE.Method.FASTTEXT):
    Utils.init_logging('Pipeline_SLC'+ str(senselabeled_or_text)+'.log')
    if do_reset:
        reset(senselabeled_or_text)

    if senselabeled_or_text: # sense-labeled corpus
        inputdata_folder = os.path.join(F.FOLDER_INPUT, F.FOLDER_SENSELABELED)
        vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, F.FOLDER_SENSELABELED)
        xml_fnames = ['semcor.xml']  # , , 'subset_omsti_aa.xml']#, 'raganato_ALL.xml', 'wngt.xml']
        SLC.organize_splits(xml_fnames)
        min_freq_vocab = 2
        lowercasing = True
    else:
        inputdata_folder = os.path.join(F.FOLDER_INPUT, F.FOLDER_STANDARDTEXT)
        vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, F.FOLDER_STANDARDTEXT)
        min_freq_vocab = 1
        lowercasing = False

    vocabulary_h5 = os.path.join(vocabulary_folder, F.VOCABULARY_OF_GLOBALS_FILENAME)

    textcorpus_fpaths = os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_STANDARDTEXT,
                                     F.FOLDER_TRAIN, F.WT_TRAIN_FILE) # used when processing WikiText-2 / standard text
    vocabulary_df = V.get_vocabulary_df(senselabeled_or_text=senselabeled_or_text, textcorpus_fpaths=[textcorpus_fpaths],
                                     vocabulary_h5_filepath=vocabulary_h5, min_count=min_freq_vocab, lowercase=lowercasing)

    if compute_single_prototype:
        reset_embeddings(senselabeled_or_text)
        single_prototypes_fname = F.SPVs_FASTTEXT_FILE if sp_method==CE.Method.FASTTEXT else F.SPVs_DISTILBERT_FILE
        single_prototypes_fpath = os.path.join(inputdata_folder, single_prototypes_fname)
        CE.compute_single_prototype_embeddings(vocabulary_df,
                                               single_prototypes_fpath,
                                               sp_method)


    vocabulary_ls = RID.retrieve_data_WordNet(vocabulary_df, inputdata_folder, vocabulary_folder)
    logging.info("CreateGraphInput.exe() > "
                 + " number of ords included in the vocabulary chunk, to be prepared: " + str(len(vocabulary_ls)))
    PI.prepare(vocabulary_ls, inputdata_folder, vocabulary_folder, embeddings_method=sp_method)
    tables.file._open_files.close_all()