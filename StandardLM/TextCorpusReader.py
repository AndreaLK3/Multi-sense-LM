import Filesystem
import Utils
import Filesystem as F
import VocabularyAndEmbeddings.Vocabulary as V
import VocabularyAndEmbeddings.Vocabulary_Utilities as VocabUtils
import os
import logging
import torch

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
                                            + "_withVocabFrom_" + "_".join(vocabulary_sources_ls))
    if corpus_name.lower() == F.WT103.lower():
        txt_corpus_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT103], split_fname)
        numIDs_outfile_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT103], split_fname + F.CORPUS_NUMERICAL_EXTENSION
                                            + "_withVocabFrom_" + "_".join(vocabulary_sources_ls))
    return txt_corpus_fpath, numIDs_outfile_fpath

# Read a corpus from a text file, convert the tokens into their numerical IDs, save the result
# The conversion relies on a vocabulary made from the specified sources, with default min_count values
def read_txt_corpus(corpus_name, split, vocabulary_sources_ls, lowercase):
    Utils.init_logging("TextCorpusReader-read_txt_corpus.log")

    txt_corpus_fpath, numIDs_outfile_fpath = get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls)
    vocab_df = V.get_vocabulary_df(vocabulary_sources_ls, lowercase, txt_min_count=1, slc_min_count=2)
    words_ls = list(vocab_df["word"])
    numerical_IDs_ls = []

    tot_tokens=0
    with open(txt_corpus_fpath, "r", encoding="utf-8") as txt_corpus_file:
        for i,line in enumerate(txt_corpus_file):
            # words = line.split() + ['<eos>']
            line_tokens, tot_tokens = VocabUtils.process_line(line, tot_tokens)
            line_tokens = list(map(lambda word_token:
                                   VocabUtils.process_word_token({'surface_form':word_token}, lowercase), line_tokens))
            line_tokens_IDs = list(map(lambda tok: words_ls.index(tok), line_tokens))
            numerical_IDs_ls.extend(line_tokens_IDs)
            logging.info("Reading and encoding the corpus. " + str(tot_tokens) + ' tokens processed...')

    numerical_IDs_t = torch.tensor(numerical_IDs_ls, dtype=torch.int64)
    torch.save(numerical_IDs_t, numIDs_outfile_fpath)


def load_corpus_IDs(corpus_name, split, vocabulary_sources_ls):
    txt_corpus_fpath, numIDs_outfile_fpath = get_corpus_fpaths(corpus_name, split, vocabulary_sources_ls)
    numerical_IDs_t = torch.load(numIDs_outfile_fpath)
    logging.info("Loaded the encoded corpus at " + str(numIDs_outfile_fpath))
    return numerical_IDs_t

