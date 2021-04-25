import Utils
import Filesystem as F
import VocabularyAndEmbeddings.Vocabulary as V
import VocabularyAndEmbeddings.Vocabulary_Utilities as VocabUtils
import os
import logging
import numpy as np
import SenseLabeledCorpus as SLC
from Models.TrainingSetup import get_objects, setup_corpus
from VocabularyAndEmbeddings.ComputeEmbeddings import SpMethod

# Auxiliary function:
# Input: corpus and split
# Outcome: get the filepaths of the text file and the numerical pre-encoding
def get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls):

    if corpus_name.lower() == F.WT2.lower():
        if split == Utils.TRAINING:
            split_fname = F.WT_TRAIN_FILE
        elif split == Utils.VALIDATION:
            split_fname = F.WT_VALID_FILE
        elif split == Utils.TEST:
            split_fname = F.WT_TEST_FILE
        txt_corpus_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT2], split_fname)
        numIDs_outfile_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT2], split_fname + F.CORPUS_NUMERICAL_EXTENSION
                                            + "withVocabFrom_" + "_".join(vocabulary_sources_ls) + ".npy")

    if corpus_name.lower() == F.SEMCOR.lower():
        if split == Utils.TRAINING:
            split_dirname = F.FOLDER_TRAIN
        elif split == Utils.VALIDATION:
            split_dirname = F.FOLDER_VALIDATION
        elif split == Utils.TEST:
            split_dirname = F.FOLDER_TEST
        txt_corpus_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], split_dirname)
        numIDs_outfile_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], split_dirname + F.CORPUS_NUMERICAL_EXTENSION
                                            + "withVocabFrom_" + "_".join(vocabulary_sources_ls) + ".npy")
    return txt_corpus_fpath, numIDs_outfile_fpath


# Top-level function:
# Read a corpus from a text file, convert the tokens into their numerical IDs, save the result
# The conversion relies on a vocabulary made from the specified sources, with default min_count values
def encode_txt_corpus(corpus_name, split, vocabulary_sources_ls, lowercase=False):
    Utils.init_logging("TextCorpusReader-read_txt_corpus.log")

    txt_corpus_fpath, numIDs_outfile_fpath = get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls)
    vocab_df = V.get_vocabulary_df(vocabulary_sources_ls, lowercase, txt_min_count=1, slc_min_count=2)
    words_ls = list(vocab_df["word"])
    logging.info("Length of the list of words from the vocabulary: " + str(len(words_ls)))
    numerical_IDs_ls = []

    tot_tokens=0
    with open(txt_corpus_fpath, "r", encoding="utf-8") as txt_corpus_file:
        for i,line in enumerate(txt_corpus_file):
            # words = line.split() + ['<eos>']
            line_tokens, tot_tokens = VocabUtils.process_line(line, tot_tokens)
            line_tokens = list(map(lambda word_token:
                                   VocabUtils.process_word_token({'surface_form':word_token}, lowercase), line_tokens))
            line_tokens_IDs = list(map(lambda tok: words_ls.index(tok), line_tokens))
            logging.debug("line_tokens_IDs=" + str(line_tokens_IDs))
            numerical_IDs_ls.extend(line_tokens_IDs)
            if (i%500) == 0:
                logging.info("Reading and encoding the corpus. " + str(tot_tokens) + ' tokens processed...')

    numerical_IDs_arr = np.array(numerical_IDs_ls)
    np.save(numIDs_outfile_fpath, numerical_IDs_arr)

# Top-level function:
# Read a sense-labeled corpus in UFSAC format, convert the tokens into their numerical IDs, save the result
def encode_slc_corpus(corpus_name, split, vocabulary_sources_ls=[F.WT2, F.SEMCOR]):
    slc_dir_fpath, numIDs_outfile_fpath = get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls)
    sp_method = SpMethod.FASTTEXT
    gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocabulary_sources_ls)

    objects = get_objects(vocabulary_sources_ls, sp_method=SpMethod.FASTTEXT, grapharea_size=32)
    dataset, dataloader = setup_corpus(objects, slc_dir_fpath, slc_or_text=True, gr_in_voc_folders= gr_in_voc_folders,
                                       batch_size=1, seq_len=1)
    dataiter = iter(dataloader)
    while True:
        ((global_forwardinput_triple, sense_forwardinput_triple), next_token_tpl) = dataiter.__next__()
        logging.info(global_forwardinput_triple)


# Auxiliary function:
# Input: Corpus name, split, sources used to create the vocabulary that we wish to use
# Outcome: load the numpy pre-encoded data
def load_corpus_IDs(corpus_name, split, vocabulary_sources_ls):
    txt_corpus_fpath, numIDs_outfile_fpath = get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls)
    numerical_IDs_t = np.load(numIDs_outfile_fpath)
    logging.info("Loaded the encoded corpus at " + str(numIDs_outfile_fpath))
    return numerical_IDs_t
