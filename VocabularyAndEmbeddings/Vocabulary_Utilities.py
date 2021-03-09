import html
import os
import logging
import nltk
import Utils
import pandas as pd
import re

# Several functions here are currently unused, and thus commented out.
# They may still be useful if we wish to do some pre-processing, so they are left in.

# Post-processing function:
# Modifies an already created vocabulary dictionary. Words are removed if frequency < min_count
def eliminate_rare_words(vocabulary_dict, min_count):
    logging.info('Removing from the vocabulary words with frequency < ' + str(min_count))
    all_words = list(vocabulary_dict.keys()) # if we operate directly on keys(), we get: RuntimeError: dictionary changed size during iteration
    for key in all_words:
        if vocabulary_dict[key][0] < min_count:
            vocabulary_dict.pop(key)

# ######### When we build a vocabulary from text, to transform the text of a line:
#
# ##### Step 0:
##### It is necessary to reconvert some of the symbols found in the WikiText datasets, in particular:
##### ‘@-@’	4 ‘@.@’ 5 metres. Possibly including the title signs, such as in ' = = = Title = = = \n'
def convert_symbols(line_text):
    symbol_patterns_ls = ['@-@', '@,@', '@.@', '@_@']
    for pat in symbol_patterns_ls:
        line_text = re.sub(pat, pat[1], line_text) # keep the symbol
    #title_pattern = ' (= ){2,}|'
    #line_text = re.sub(title_pattern, "", line_text)
    return line_text
#
#
##### Step 1: recreate <unk> from '<','unk','>'
def restore_unk_token(list_of_tokens):
    modified_tokens = []
    i = 0
    while i < (len(list_of_tokens)):
        tok = list_of_tokens[i]
        try: # try & except, in case the corpus ends with a <
            if tok == '<' and list_of_tokens[i+1]=='unk' and list_of_tokens[i+2] == '>':
                    modified_tokens.append(''.join(list_of_tokens[i:i+3])) #include <unk>
                    i = i+3 #and go beyond it
            else:
                modified_tokens.append(tok)
                i = i + 1
        except IndexError:
            return list_of_tokens
    return modified_tokens
#
##### Step 2: There are word-tokens with non-Latin and non-Greek characters, such as: 匹, 枚, マギカ , etc.
##### We replace those words with <unk>
# def replace_nonLatinGreek_words(list_of_tokens):
#     modified_tokens = []
#     for tok in list_of_tokens:
#         try:
#             tok.encode('latin-1')
#             modified_tokens.append(tok)
#         except UnicodeEncodeError:
#             try:
#                 tok.encode('greek')
#                 modified_tokens.append(tok)
#             except UnicodeEncodeError:
#                 modified_tokens.append(Utils.UNK_TOKEN)
#     return modified_tokens
#
#
##### Step 3: Numbers. All decimals, together with all numbers that are not 1 digit (basic) or 4 digits (years),
##### get replaced with <num>
# def replace_numbers(list_of_tokens):
#     modified_tokens = []
#     nums_patterns = [re.compile('[0-9]+[.,][0-9]+'), re.compile('([0-9]){5,}'), re.compile('([0-9]){2,3}')]
#     for tok in list_of_tokens:
#         matches = [re.match(nums_pattern, tok) for nums_pattern in nums_patterns]
#         if matches == [None, None, None]:
#             modified_tokens.append(tok)
#         else:
#             valid_matches = list(filter(lambda m : m is not None , matches))
#             matches_sorted_by_length = sorted(valid_matches, key= lambda m : len(m.group()), reverse=True)
#             longest_match = matches_sorted_by_length[0].group()
#             if len(longest_match) == len(tok): # replace with <NUM>
#                 modified_tokens.append(Utils.NUM_TOKEN)
#             else:  # the number is only a part of the token. as normal
#                 modified_tokens.append(tok)
#
#     return modified_tokens
#
##### Process one line: tokenize, convert symbols like @-@, manage <unk>, replace numbers etc.
def process_line(line, tot_tokens=0):

      no_ampersand_symbols_line = convert_symbols(line)  # eliminate @-@, @.@ etc.
      raw_sentences_ls = nltk.tokenize.sent_tokenize(no_ampersand_symbols_line) # split line into sentences at the periods

      sentences_ls = list(map(lambda sent: sent.strip(), raw_sentences_ls))  # eliminate space at start and end
#     truecased_sentences_ls = list(map(lambda sent: sent[0:1].lower() + sent[1:], sentences_ls)) #true-casing

      tokenized_sentences_lls = list(map(lambda sent: nltk.tokenize.word_tokenize(sent), sentences_ls))
      line_tokens_00 = [elem for ls in tokenized_sentences_lls for elem in ls] # raw tokens
      tot_tokens = tot_tokens + len(line_tokens_00)

      line_tokens_01_unk = restore_unk_token(line_tokens_00) # recreate <unk> from '<','unk','>'
# #     line_tokens_02_onlyLatinOrGreek = replace_nonLatinGreek_words(line_tokens_01_unk)
# #     line_tokens_03_replNumbers = replace_numbers(line_tokens_02_onlyLatinOrGreek)

      return line_tokens_01_unk, tot_tokens


######### When we build a vocabulary from the sense-labeled corpus(es), to transform a token:

def process_word_token(token_dict, lowercasing):
    # ------- the only essential step
    token_text = html.unescape(str(token_dict['surface_form']))

    if lowercasing:
        token_text = token_text.lower()

    # ------- keeping _ in tokens in SemCor, e.g. Fulton_County
    # token_text = token_text.replace('_', ' ')  # we keep phrases, but we should write 'Mr. Barcus' not Mr._Barcus
    #
    # # ------- superfluous on not-WT2
    # token_latinorgreek = replace_nonLatinGreek_words([token_text])[0]

    return token_text

