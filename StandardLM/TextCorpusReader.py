import Filesystem
import Utils
import Filesystem as F
import VocabularyAndEmbeddings.Vocabulary as V
import VocabularyAndEmbeddings.Vocabulary_Utilities as VocabUtils
import os
import logging
import torch
import numpy as np


# Auxiliary function:
# Input: corpus and split
# Outcome: get the filepaths of the text file and the numerical pre-encoding
def get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls):
    if split == Utils.TRAINING:
        split_fname = F.WT_TRAIN_FILE
    elif split == Utils.VALIDATION:
        split_fname = F.WT_VALID_FILE
    elif split == Utils.TEST:
        split_fname = F.WT_TEST_FILE

    if corpus_name.lower() == F.WT2.lower():
        txt_corpus_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT2], split_fname)
        numIDs_outfile_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT2], split_fname + F.CORPUS_NUMERICAL_EXTENSION
                                            + "withVocabFrom_" + "_".join(vocabulary_sources_ls) + ".npy")
    if corpus_name.lower() == F.WT103.lower():
        txt_corpus_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT103], split_fname)
        numIDs_outfile_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT103], split_fname + F.CORPUS_NUMERICAL_EXTENSION
                                            + "withVocabFrom_" + "_".join(vocabulary_sources_ls) + ".npy")
    if corpus_name.lower() == F.SEMCOR.lower():
        txt_corpus_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], split_fname)
        numIDs_outfile_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], split_fname + F.CORPUS_NUMERICAL_EXTENSION
                                            + "withVocabFrom_" + "_".join(vocabulary_sources_ls) + ".npy")
    return txt_corpus_fpath, numIDs_outfile_fpath


# Top-level function:
# Read a corpus from a text file, convert the tokens into their numerical IDs, save the result
# The conversion relies on a vocabulary made from the specified sources, with default min_count values
def read_txt_corpus(corpus_name, split, vocabulary_sources_ls, lowercase):
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

# Auxiliary function:
# Input: Corpus name, split, sources used to create the vocabulary that we wish to use
# Outcome: load the numpy pre-encoded data
def load_corpus_IDs(corpus_name, split, vocabulary_sources_ls):
    txt_corpus_fpath, numIDs_outfile_fpath = get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls)
    numerical_IDs_t = np.load(numIDs_outfile_fpath)
    logging.info("Loaded the encoded corpus at " + str(numIDs_outfile_fpath))
    return numerical_IDs_t



# Another top-level function:
# Input: None. It operates on the sources we wish to use (WT2 corpus, vocabulary from WT2 + SemCor)
# Outcome: It eliminates all old versions the encoded splits, and creates them
def reset_vocab_and_splits_wt2plus(sp_method=Utils.SpMethod.FASTTEXT):
    vocab_sources_ls = [F.WT2, F.SEMCOR]
    _, _, vocab_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)

    _, encoded_train_split_fpath = get_corpus_fpaths(corpus_name=F.WT2, split=Utils.TRAINING,
                                                     vocabulary_sources_ls=vocab_sources_ls)
    _, encoded_valid_split_fpath = get_corpus_fpaths(corpus_name=F.WT2, split=Utils.VALIDATION,
                                                     vocabulary_sources_ls=vocab_sources_ls)
    _, encoded_test_split_fpath = get_corpus_fpaths(corpus_name=F.WT2, split=Utils.TEST,
                                                     vocabulary_sources_ls=vocab_sources_ls)
    all_fpaths = [encoded_train_split_fpath, encoded_valid_split_fpath, encoded_test_split_fpath]

    for fpath in all_fpaths:
        if os.path.exists(fpath):
            os.remove(fpath)

    V.get_vocabulary_df(corpora_names=vocab_sources_ls, lowercase=False)
    read_txt_corpus(corpus_name=F.WT2, split=Utils.TRAINING, vocabulary_sources_ls=vocab_sources_ls, lowercase=False)
    read_txt_corpus(corpus_name=F.WT2, split=Utils.VALIDATION, vocabulary_sources_ls=vocab_sources_ls, lowercase=False)
    read_txt_corpus(corpus_name=F.WT2, split=Utils.TEST, vocabulary_sources_ls=vocab_sources_ls, lowercase=False)
