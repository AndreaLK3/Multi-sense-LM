import re
import string
import time
import os
import logging
import nltk
import Utils
import pandas as pd
import re

def get_vocabulary_df(vocabulary_h5_filepath, corpus_txt_filepath, min_count, extended_lang_id):
    if os.path.exists(vocabulary_h5_filepath):
        vocab_df = pd.read_hdf(vocabulary_h5_filepath, mode='r')
    else:
        vocabulary_h5 = pd.HDFStore(vocabulary_h5_filepath, mode='w')
        vocab_h5_itemsizes = {'word': Utils.HDF5_BASE_CHARSIZE / 4, 'frequency': Utils.HDF5_BASE_CHARSIZE / 8}

        vocabulary = build_vocabulary_from_corpus(corpus_txt_filepath, extended_lang_id)
        eliminate_rare_words(vocabulary, min_count)
        vocab_df = pd.DataFrame(data=zip(vocabulary.keys(), vocabulary.values()), columns=['word', 'frequency'])
        vocabulary_h5.append(key='vocabulary', value=vocab_df, min_itemsize=vocab_h5_itemsizes)
        vocabulary_h5.close()

    return vocab_df


def eliminate_short_numbers(list_of_tokens):
    modified_tokens = []
    nums_pattern = re.compile('([0-9]){1,3}')
    for tok in list_of_tokens:
        match = re.match(nums_pattern, tok)
        if match is None:
            modified_tokens.append(tok)
        else:
            part = match.group()
            if len(part) == len(tok): # replace with <NUM>
                modified_tokens.append(Utils.NUM_TOKEN)
            else:  # the number is only a part of the token. as normal
                modified_tokens.append(tok)

    return modified_tokens

# modifies the dictionary. Words are removed if frequency < min_count
def eliminate_rare_words(vocabulary_dict, min_count):
    logging.info('Removing from the vocabulary words with frequency < ' + str(min_count))
    all_words = list(vocabulary_dict.keys()) # if we operate directly on keys(), we get: RuntimeError: dictionary changed size during iteration
    for key in all_words:
        if vocabulary_dict[key] < min_count:
            vocabulary_dict.pop(key)


def build_vocabulary_from_corpus(corpus_txt_filepath, extended_lang_id):

    vocab_dict = {}
    tot_tokens = 0
    stopwords_ls = nltk.corpus.stopwords.words(extended_lang_id)

    time_prev = time.time()
    for i, line in enumerate(open(corpus_txt_filepath, "r", encoding="utf-8")):
        if line == '':
            break
        # tokens_in_line = nltk.tokenize.word_tokenize(line)
        line_noPuncts = re.sub('[' + string.punctuation.replace('-', '') + ']', ' ', line)
        tokens_in_line_nopuncts = nltk.tokenize.word_tokenize(line_noPuncts)
        tokens_in_line_nonumbers = eliminate_short_numbers(tokens_in_line_nopuncts)
        tokens_in_line_lowercase = list(map(lambda tok : tok.lower(), tokens_in_line_nonumbers))
        tokens_in_line_nostopwords = list(filter(lambda tok: tok not in stopwords_ls, tokens_in_line_lowercase))

        tot_tokens = tot_tokens + len(tokens_in_line_nostopwords)

        if i % 10000 == 0:
            time_next = time.time()
            time_elapsed = round( time_next - time_prev, 4)
            print("Reading in line n. : " + str(i) + ' ; number of tokens encountered: ' + str(tot_tokens) +
                  " ; time elapsed = " + str(time_elapsed) + " s")
            time_prev = time.time()

        different_tokens = set(token for token in tokens_in_line_lowercase)

        update_lts = [(token, line.count(token) ) for token in different_tokens]
        for word, freq in update_lts:
            try:
                prev_freq = vocab_dict[word]
                vocab_dict[word] = prev_freq + freq
            except KeyError:
                vocab_dict[word] = freq

    logging.info("Vocabulary dictionary created, after processing " + str(tot_tokens) + ' tokens')
    return vocab_dict