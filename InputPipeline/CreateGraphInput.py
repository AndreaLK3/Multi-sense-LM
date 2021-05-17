import Filesystem
import Filesystem as F
import GetKBData.RetrieveInputData as RID
import GetKBData.PrepareKBInput as PI
import Utils
import os
import pandas as pd
import VocabularyAndEmbeddings.Vocabulary as V
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import tables
import logging
import SenseLabeledCorpus as SLC
from time import time

# Before starting: clean all storage files; reset vocabulary index to 0
def reset(vocabulary_sources_ls, sp_method=Utils.SpMethod.FASTTEXT):

    graph_folder, inputdata_folder, vocab_folder = F.get_folders_graph_input_vocabulary(vocabulary_sources_ls, sp_method)

    vocab_h5_fname = "vocabulary_" + "_".join(vocabulary_sources_ls) + ".h5"
    vocab_txt_fname = vocab_h5_fname.replace(".h5", ".txt")
    vocab_filepaths = [os.path.join(vocab_folder, vocab_h5_fname), os.path.join(vocab_folder, vocab_txt_fname)]

    # reset the hdf5 archives for dictionary information: definitions, examples, synonyms, antonyms
    archives_filenames = list(filter(lambda fname: 'h5' in fname, os.listdir(inputdata_folder)))
    archives_filepaths = list(map(lambda fname: os.path.join(inputdata_folder, fname), archives_filenames))
    for fpath in archives_filepaths:
        f = pd.HDFStore(fpath, mode='w')
        f.close()

    # The SQL DB with the indices for the embedding matrices
    db_filepaths = [os.path.join(inputdata_folder, Utils.INDICES_TABLE_DB)]

    # reset the graph object file, and the area_matrices
    graph_filepaths = [os.path.join(graph_folder, F.KBGRAPH_FILE)] + \
                      [os.path.join(graph_folder, fname) for fname in os.listdir(F.FOLDER_GRAPH) if '.npz' in fname]

    for fpath in vocab_filepaths + db_filepaths + graph_filepaths:
        if os.path.exists(fpath):
            os.remove(fpath)

    with open(os.path.join(vocab_folder, F.VOCAB_CURRENT_INDEX_FILE), 'w') as vi_file:
        vi_file.write("0")
        vi_file.close()


def reset_embeddings(vocabulary_sources_ls, sp_method=Utils.SpMethod.FASTTEXT):
    inputdata_folder = Filesystem.set_directory(os.path.join(F.FOLDER_INPUT, "_".join(vocabulary_sources_ls), sp_method.value))
    # reset the embeddings, both those for dictionary elements and those for single-prototype vectors
    vectorized_inputs_filenames = list(filter(lambda fname: '.npy' in fname, os.listdir(inputdata_folder)))
    vectorized_inputs_filepaths = list(map(lambda fname: os.path.join(inputdata_folder, fname),
                                           vectorized_inputs_filenames))
    for fpath in vectorized_inputs_filepaths:
        if os.path.exists(fpath):
            os.remove(fpath)



def exe_from_input_to_vectors(do_reset, compute_single_prototype, vocabulary_sources_ls, sp_method=Utils.SpMethod.FASTTEXT):
    Utils.init_logging("CGI-pipelineStart.log")

    t0 = time()
    if do_reset:
        reset(vocabulary_sources_ls, sp_method)

    # set the folder that will contain the WordNet gloss data, depending on vocabulary sources and embeddings method
    inputdata_folder = Filesystem.set_directory(os.path.join(F.FOLDER_INPUT, "_".join(vocabulary_sources_ls), sp_method.value))

    # organize the SemCor corpus in training, validation and test splits
    SLC.organize_splits(xml_folder_path=F.CORPORA_LOCATIONS[F.SEMCOR], xml_fname='semcor.xml')

    # create the Vocabulary from the specified sources
    vocabulary_df = V.get_vocabulary_df(corpora_names=vocabulary_sources_ls, lowercase=False, slc_min_count=2, txt_min_count=1)
    t1 = time()
    logging.info("Old files deleted, vocabulary created. Time elapsed="+str(round(t1-t0,2)))

    if compute_single_prototype:
        reset_embeddings(vocabulary_sources_ls, sp_method)
        single_prototypes_fpath = os.path.join(inputdata_folder, F.SPVs_FILENAME)
        CE.compute_single_prototype_embeddings(vocabulary_df,
                                               single_prototypes_fpath,
                                               sp_method)

    vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, "_".join(vocabulary_sources_ls))
    vocabulary_ls = RID.retrieve_data_WordNet(vocabulary_df, inputdata_folder, vocabulary_folder)
    t2 = time()
    logging.info("Did retrieve data from WordNet. Time elapsed= " + str(round(t2-t1,2)))

    PI.prepare(vocabulary_ls, inputdata_folder, vocabulary_folder, embeddings_method=sp_method)
    tables.file._open_files.close_all()
    t3 = time()
    logging.info("Did preprocessing (no duplicate glosses + senses'table + sentence encoding)=" + str(round(t3-t2,2)))