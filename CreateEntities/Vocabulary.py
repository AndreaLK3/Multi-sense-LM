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
        logging.info("*** The vocabulary was loaded from the file " + vocabulary_h5_filepath)
    else:
        vocabulary_h5 = pd.HDFStore(vocabulary_h5_filepath, mode='w')
        vocab_h5_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'frequency': Utils.HDF5_BASE_SIZE_512 / 8}

        vocabulary = build_vocabulary_from_corpus(corpus_txt_filepath, extended_lang_id)
        eliminate_rare_words(vocabulary, min_count)
        vocab_df = pd.DataFrame(data=zip(vocabulary.keys(), vocabulary.values()), columns=['word', 'frequency'])
        vocabulary_h5.append(key='vocabulary', value=vocab_df, min_itemsize=vocab_h5_itemsizes)
        vocabulary_h5.close()
        logging.info("*** The vocabulary was created from the corpus file " + corpus_txt_filepath)

    return vocab_df


# To transform a line

def replace_numbers(list_of_tokens):
    modified_tokens = []
    nums_pattern = re.compile('([0-9]){5,}|([0-9]){1,3}')
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


# There are strings with non-latin characters, such as: 匹, 枚, マギカ , etc.
# In those, we replace the non-latin words with <unk>
def replace_nonLatin_words(list_of_tokens):
    modified_tokens = []
    for tok in list_of_tokens:
        try:
            tok.encode('latin-1')
            modified_tokens.append(tok)
        except UnicodeEncodeError:
            modified_tokens.append(Utils.UNK_TOKEN)
    return modified_tokens


# modifies an already created vocabulary dictionary. Words are removed if frequency < min_count
def eliminate_rare_words(vocabulary_dict, min_count):
    logging.info('Removing from the vocabulary words with frequency < ' + str(min_count))
    all_words = list(vocabulary_dict.keys()) # if we operate directly on keys(), we get: RuntimeError: dictionary changed size during iteration
    for key in all_words:
        if vocabulary_dict[key] < min_count:
            vocabulary_dict.pop(key)

# keeps the <unk> token
def remove_punctuation (list_of_tokens):
    modified_tokens = []
    i = 0
    while i < (len(list_of_tokens)):
        tok = list_of_tokens[i]
        if tok not in string.punctuation:
            modified_tokens.append(tok)
            i = i+1
        else:
            if tok == '<' and list_of_tokens[i+1]==['unk'] and list_of_tokens[i+2] == '>':
                modified_tokens.extend(list_of_tokens[i:i+3])
                i = i+3 #include <unk>
            else:
                i = i+1 #skip punctuation
    return modified_tokens



def process_line(line, stopwords_ls, tot_tokens=0):
    line_noperiods = re.sub('[.]', ' ', line)
    tokens_in_line = nltk.tokenize.word_tokenize(line_noperiods)
    tot_tokens = tot_tokens + len(tokens_in_line)
    tokens_in_line_nopuncts = remove_punctuation(tokens_in_line)
    tokens_in_line_nostopwords = list(filter(lambda tok: tok not in stopwords_ls, tokens_in_line_nopuncts))
    tokens_in_line_latinWords = replace_nonLatin_words(tokens_in_line_nostopwords)
    tokens_in_line_nonumbers = replace_numbers(tokens_in_line_latinWords)

    tokens_in_line_lowercase = list(map(lambda tok: tok.lower(), tokens_in_line_nonumbers))
    return tokens_in_line_lowercase, tot_tokens


def build_vocabulary_from_corpus(corpus_txt_filepath, extended_lang_id):

    vocab_dict = {}
    tot_tokens = 0
    stopwords_ls = nltk.corpus.stopwords.words(extended_lang_id)
    time_prev = time.time()

    for i, line in enumerate(open(corpus_txt_filepath, "r", encoding="utf-8")):
        if line == '':
            break
        tokens_in_line_lowercase, tot_tokens = process_line(line, stopwords_ls, tot_tokens)

        if i % 10000 == 0:
            time_next = time.time()
            time_elapsed = round( time_next - time_prev, 4)
            logging.info("Reading in line n. : " + str(i) + ' ; number of tokens encountered: ' + str(tot_tokens) +
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

    logging.info("Vocabulary created, after processing " + str(tot_tokens) + ' tokens')
    return vocab_dict