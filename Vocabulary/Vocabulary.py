import string
import Utils
import Filesystem as F
import os
import pandas as pd
import GNN.SenseLabeledCorpus as SLC
import logging
import Vocabulary.Vocabulary_Utilities as VocabUtils


def build_vocabulary_from_text(corpus_txt_filepath):
    vocab_dict = {}
    tot_tokens = 0

    for i, line in enumerate(open(corpus_txt_filepath, "r", encoding="utf-8")):
        if line == '':
            break
        tokens_in_line_lowercase, tot_tokens = VocabUtils.process_line(line, tot_tokens)

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




def build_vocabulary_from_senselabeled(slc_split_name):
    vocab_dict = {}

    tokens_toexclude = [Utils.EOS_TOKEN] # + list(string.punctuation)
    # Commas and punctuation signs are present in the Sense-Labeled Corpus as separate tokens.
    # Therefore, it makes sense to keep them in the vocabulary, and thus in the graph, as globals

    for token_dict in SLC.read_split(slc_split_name):
        token = VocabUtils.process_slc_token(token_dict)
        if token not in tokens_toexclude:
            try:
                prev_freq = vocab_dict[token]
                vocab_dict[token] = prev_freq + 1
            except KeyError:
                vocab_dict[token] = 1
    if Utils.UNK_TOKEN not in vocab_dict.keys():
        vocab_dict[Utils.UNK_TOKEN] = 100 # add it manually
        return vocab_dict


# Entry function: if a vocabulary is already present in the specified path, load it. Otherwise, create it.
def get_vocabulary_df(senselabeled_or_text, slc_split_name, corpus_txt_filepath, out_vocabulary_h5_filepath, min_count):
    if os.path.exists(out_vocabulary_h5_filepath):
        vocab_df = pd.read_hdf(out_vocabulary_h5_filepath, mode='r')
        logging.info("*** The vocabulary was loaded from the file " + out_vocabulary_h5_filepath)
    else:
        logging.info("*** Creating vocabulary at " + out_vocabulary_h5_filepath)
        vocabulary_h5 = pd.HDFStore(out_vocabulary_h5_filepath, mode='w')
        vocab_h5_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'frequency': Utils.HDF5_BASE_SIZE_512 / 8}

        if senselabeled_or_text:
            vocabulary = build_vocabulary_from_senselabeled(slc_split_name)
        else:
            vocabulary = build_vocabulary_from_text(corpus_txt_filepath)

        VocabUtils.eliminate_rare_words(vocabulary, min_count)

        vocab_df = pd.DataFrame(data=zip(vocabulary.keys(), vocabulary.values()), columns=['word', 'frequency'])
        vocabulary_h5.append(key='vocabulary', value=vocab_df, min_itemsize=vocab_h5_itemsizes)
        vocabulary_h5.close()
        logging.info("*** The vocabulary was created from the " + slc_split_name + " split of the sense-labeled corpus ")

    return vocab_df