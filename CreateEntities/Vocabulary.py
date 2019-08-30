import re
import string
import time
import os
import gensim.models
import nltk
import Utils
import pandas as pd


def get_vocabulary_df(vocabulary_h5_filepath, corpus_txt_filepath):
    if os.path.exists(vocabulary_h5_filepath):
        vocab_df = pd.read_hdf(vocabulary_h5_filepath, mode='r')
    else:
        vocabulary_h5 = pd.HDFStore(vocabulary_h5_filepath, mode='w')
        vocab_h5_itemsizes = {'word': Utils.HDF5_BASE_CHARSIZE / 4, 'frequency': Utils.HDF5_BASE_CHARSIZE / 8}

        vocabulary = build_vocabulary_from_corpus(corpus_txt_filepath)
        vocab_df = pd.DataFrame(data=zip(vocabulary.keys(), vocabulary.values()), columns=['word', 'frequency'])
        vocabulary_h5.append(key='vocabulary', value=vocab_df, min_itemsize=vocab_h5_itemsizes)

    return vocab_df


def build_vocabulary_from_corpus(corpus_txt_filepath):

    vocab_dict = {}
    tot_tokens = 0

    time_prev = time.time()
    for i, line in enumerate(open(corpus_txt_filepath, "r", encoding="utf-8")):
        if line == '':
            break
        # tokens_in_line = nltk.tokenize.word_tokenize(line)
        line_noPuncts = re.sub('[' + string.punctuation.replace('-', '') + ']', ' ', line)
        tokens_in_line = nltk.tokenize.word_tokenize(line_noPuncts)
        tot_tokens = tot_tokens + len(tokens_in_line)

        if i % 2000 == 0:
            time_next = time.time()
            time_elapsed = round( time_next - time_prev, 4)
            print("Reading in line n. : " + str(i) + ' ; number of tokens encountered: ' + str(tot_tokens) +
                  " ; time elapsed = " + str(time_elapsed) + " s")
            time_prev = time.time()

        different_tokens = set(token for token in tokens_in_line)

        update_lts = [(token, line.count(token) ) for token in different_tokens]
        for word, freq in update_lts:
            try:
                prev_freq = vocab_dict[word]
                vocab_dict[word] = prev_freq + freq
            except KeyError:
                vocab_dict[word] = freq

    return vocab_dict