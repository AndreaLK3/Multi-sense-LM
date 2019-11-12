import time
import os
import logging
import nltk
import Utils
import pandas as pd
import re



# Post-processing function:
# Modifies an already created vocabulary dictionary. Words are removed if frequency < min_count
def eliminate_rare_words(vocabulary_dict, min_count):
    logging.info('Removing from the vocabulary words with frequency < ' + str(min_count))
    all_words = list(vocabulary_dict.keys()) # if we operate directly on keys(), we get: RuntimeError: dictionary changed size during iteration
    for key in all_words:
        if vocabulary_dict[key] < min_count:
            vocabulary_dict.pop(key)


######### To transform the text of a line:

##### Step 0:
##### It is necessary to reconvert some of the symbols found in the WikiText datasets, in particular:
##### ‘@-@’	4 ‘@.@’ 5 metres. Including also the title signs, such as in ' = = = Title = = = \n'
def convert_symbols(line_text):
    symbol_patterns_ls = [' @-@ ', ' @,@ ', ' @.@ ', ' @_@ ']
    for pat in symbol_patterns_ls:
        line_text = re.sub(pat, pat[2], line_text) #keep the symbol, and eliminate the spaces too
    title_pattern = ' (= ){2,}|'
    line_text = re.sub(title_pattern, "", line_text)
    return line_text

##### Step 1: nltk's tokenizer - sentence tokenizer

##### Step 2: Eliminate outer spaces, and true-case the sentences.

##### Step 3: Split into word tokens, and reunite the lls

##### Step 4: recreate <unk> from '<','unk','>'
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

##### Step 5: There are word-tokens with non-Latin and non-Greek characters, such as: 匹, 枚, マギカ , etc.
##### We replace those words with <unk>
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


##### Step 6: Numbers. All decimals, together with all numbers that are not 1 digit (basic) or 4 digits (years),
##### get replaced with <num>
def replace_numbers(list_of_tokens):
    modified_tokens = []
    nums_patterns = [re.compile('[0-9]+[.,][0-9]+'), re.compile('([0-9]){5,}'), re.compile('([0-9]){2,3}')]
    for tok in list_of_tokens:
        matches = [re.match(nums_pattern, tok) for nums_pattern in nums_patterns]
        if matches == [None, None, None]:
            modified_tokens.append(tok)
        else:
            valid_matches = list(filter(lambda m : m is not None , matches))
            matches_sorted_by_length = sorted(valid_matches, key= lambda m : len(m.group()), reverse=True)
            longest_match = matches_sorted_by_length[0].group()
            if len(longest_match) == len(tok): # replace with <NUM>
                modified_tokens.append(Utils.NUM_TOKEN)
            else:  # the number is only a part of the token. as normal
                modified_tokens.append(tok)

    return modified_tokens

##### Process one line: tokenize, convert symbols like @-@, manage <unk>, replace numbers etc.
def process_line(line, tot_tokens=0):

    no_ampersand_symbols_line = convert_symbols(line)  # eliminate @-@, @.@ etc.
    raw_sentences_ls = nltk.tokenize.sent_tokenize(no_ampersand_symbols_line) # split line into sentences at the periods

    sentences_ls = list(map(lambda sent: sent.strip(), raw_sentences_ls))  # eliminate space at start and end
    truecased_sentences_ls = list(map(lambda sent: sent[0:1].lower() + sent[1:], sentences_ls)) #true-casing

    tokenized_sentences_lls = list(map(lambda sent: nltk.tokenize.word_tokenize(sent), truecased_sentences_ls))
    line_tokens_00 = [elem for ls in tokenized_sentences_lls for elem in ls] # raw tokens
    tot_tokens = tot_tokens + len(line_tokens_00)

    line_tokens_01_unk = restore_unk_token(line_tokens_00) # recreate <unk> from '<','unk','>'
    line_tokens_02_onlyLatinOrGreek = replace_nonLatinGreek_words(line_tokens_01_unk)
    line_tokens_03_replNumbers = replace_numbers(line_tokens_02_onlyLatinOrGreek)

    return line_tokens_03_replNumbers, tot_tokens





def build_vocabulary_from_corpus(corpus_txt_filepath):
    #Utils.init_logging(os.path.join("PrepareGraphInput", "Vocabulary.log"), logging.INFO)
    vocab_dict = {}
    tot_tokens = 0
    time_prev = time.time()

    for i, line in enumerate(open(corpus_txt_filepath, "r", encoding="utf-8")):
        if line == '':
            break
        tokens_in_line_lowercase, tot_tokens = process_line(line, tot_tokens)

        if i % 10000 == 0:
            time_next = time.time()
            time_elapsed = round( time_next - time_prev, 4)
            logging.info("Building vocabulary. Reading in line n. : " + str(i) + ' ; number of tokens encountered: ' + str(tot_tokens) +
                  " ; time elapsed = " + str(time_elapsed) + " s")
            time_prev = time.time()

        different_tokens = set(token for token in tokens_in_line_lowercase)

        update_lts = [(token.replace('_', ' '), line.count(token) ) for token in different_tokens] # '_' was used for phrases.
        for word, freq in update_lts:
            try:
                prev_freq = vocab_dict[word]
                vocab_dict[word] = prev_freq + freq
            except KeyError:
                vocab_dict[word] = freq

    logging.info("Vocabulary created, after processing " + str(tot_tokens) + ' tokens')
    return vocab_dict


# Entry function: if a vocabulary is already present in the specified path, load it. Otherwise, create it.
def get_vocabulary_df(vocabulary_h5_filepath, corpus_txt_filepath, min_count):
    if os.path.exists(vocabulary_h5_filepath):
        vocab_df = pd.read_hdf(vocabulary_h5_filepath, mode='r')
        logging.info("*** The vocabulary was loaded from the file " + vocabulary_h5_filepath)
    else:
        vocabulary_h5 = pd.HDFStore(vocabulary_h5_filepath, mode='w')
        vocab_h5_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'frequency': Utils.HDF5_BASE_SIZE_512 / 8}

        vocabulary = build_vocabulary_from_corpus(corpus_txt_filepath)
        eliminate_rare_words(vocabulary, min_count)
        vocab_df = pd.DataFrame(data=zip(vocabulary.keys(), vocabulary.values()), columns=['word', 'frequency'])
        vocabulary_h5.append(key='vocabulary', value=vocab_df, min_itemsize=vocab_h5_itemsizes)
        vocabulary_h5.close()
        logging.info("*** The vocabulary was created from the corpus file " + corpus_txt_filepath)

    return vocab_df