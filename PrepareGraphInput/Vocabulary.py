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


# To transform the text of a line:


# It is also necessary to reconvert some of the symbols found in the WikiText datasets, in particular:
# ‘@-@’	4 ‘@.@’ 5 metres
def convert_symbols(line_text):
    patterns_ls = [' @-@ ', ' @,@ ', ' @.@ ', ' @_@ ']
    for pat in patterns_ls:
        line_text = re.sub(pat, pat[2], line_text) #keep the symbol, and eliminate the spaces too
    return line_text

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


def restore_unk_token(list_of_tokens):
    modified_tokens = []
    i = 0
    while i < (len(list_of_tokens)):
        tok = list_of_tokens[i]
        if tok == '<' and list_of_tokens[i+1]=='unk' and list_of_tokens[i+2] == '>':
                modified_tokens.append(''.join(list_of_tokens[i:i+3])) #include <unk>
                i = i+3 #and go beyond it
        else:
            modified_tokens.append(tok)
            i = i + 1
    return modified_tokens

# There are strings with non-Latin and non-Greek characters, such as: 匹, 枚, マギカ , etc.
# We replace those words with <unk>
def replace_nonLatinGreek_words(list_of_tokens):
    modified_tokens = []
    for tok in list_of_tokens:
        try:
            tok.encode('latin-1')
            modified_tokens.append(tok)
        except UnicodeEncodeError:
            try:
                tok.encode('greek')
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



def process_line(line, tot_tokens=0):
    raw_sentences_ls = nltk.tokenize.sent_tokenize(line)
    sentences_ls = list(map(lambda sent: sent.strip(), raw_sentences_ls)) # eliminate space at start and end
    # truecasing: lowercase only the word at the beginning of the sentence
    truecased_sentences_ls = list(map(lambda sent: sent[0:1].lower() + sent[1:], sentences_ls))
    tokenized_sentences_lls = list(map(lambda sent: nltk.tokenize.word_tokenize(sent), truecased_sentences_ls))
    line_tokens = [elem for ls in tokenized_sentences_lls for elem in ls]
    tot_tokens = tot_tokens + len(line_tokens)
    tokens_in_line_latinWords = replace_nonLatinGreek_words(line_tokens)
    tokens_in_line_nonumbers = replace_numbers(tokens_in_line_latinWords)

    tokens_in_line_lowercase = list(map(lambda tok: tok.lower(), tokens_in_line_nonumbers))
    return tokens_in_line_lowercase, tot_tokens


def build_vocabulary_from_corpus(corpus_txt_filepath, extended_lang_id="'english"):
    Utils.init_logging(os.path.join("PrepareGraphInput", "Vocabulary.log"), logging.INFO)
    vocab_dict = {}
    tot_tokens = 0
    time_prev = time.time()

    for i, line in enumerate(open(corpus_txt_filepath, "r", encoding="utf-8")):
        if line == '':
            break
        logging.info("\n " + str(i) + " \t - " + line)
        no_ampersand_symbols_line = convert_symbols(line) # eliminate @-@, @.@ etc.
        raw_sentences_ls = nltk.tokenize.sent_tokenize(no_ampersand_symbols_line)
        logging.info("\n No ampersand symbols:" + str(i) + " \t - " + str(raw_sentences_ls))
        sentences_ls = list(map(lambda sent: sent.strip(), raw_sentences_ls))  # eliminate space at start and end
        truecased_sentences_ls = list(map(lambda sent: sent[0:1].lower() + sent[1:], sentences_ls))
        logging.info("\n No outer spaces; truecased" + str(i) + " \t - " + str(truecased_sentences_ls))
        tokenized_sentences_lls = list(map(lambda sent: nltk.tokenize.word_tokenize(sent), truecased_sentences_ls))
        line_tokens_00 = [elem for ls in tokenized_sentences_lls for elem in ls]
        line_tokens_01_unk = restore_unk_token(line_tokens_00)
        logging.info("\n Tokenized, and then restored the <unk> token" + str(i) + " \t - " + str(line_tokens_01_unk))
        line_tokens_02_latinGreekWords = replace_nonLatinGreek_words(line_tokens_01_unk)
        logging.info("\n Tokens that are neither Latin nor Greek get replaced with <unk>" + str(i) + " \t - " + str(line_tokens_02_latinGreekWords))




    #     tokens_in_line_lowercase, tot_tokens = process_line(line, tot_tokens)
    #
    #     if i % 10000 == 0:
    #         time_next = time.time()
    #         time_elapsed = round( time_next - time_prev, 4)
    #         logging.info("Reading in line n. : " + str(i) + ' ; number of tokens encountered: ' + str(tot_tokens) +
    #               " ; time elapsed = " + str(time_elapsed) + " s")
    #         time_prev = time.time()
    #
    #     different_tokens = set(token for token in tokens_in_line_lowercase)
    #
    #     update_lts = [(token, line.count(token) ) for token in different_tokens]
    #     for word, freq in update_lts:
    #         try:
    #             prev_freq = vocab_dict[word]
    #             vocab_dict[word] = prev_freq + freq
    #         except KeyError:
    #             vocab_dict[word] = freq
    #
    # logging.info("Vocabulary created, after processing " + str(tot_tokens) + ' tokens')
    return vocab_dict