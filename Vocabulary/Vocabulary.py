import Utils
import os
import pandas as pd
import SenseLabeledCorpus as SLC
import logging
import Vocabulary.Vocabulary_Utilities as VocabUtils


def build_vocabulary_from_text(corpus_txt_fpaths):
    vocab_dict = {}
    lines_logpoint = 1000

    for corpus_txt_fpath in corpus_txt_fpaths:
        # from the method of vocabulary creation used in AWD-LSTM. We do not process or modify the lines&tokens

        # """Tokenizes a text file."""
        assert os.path.exists(corpus_txt_fpath)
        # Add words to the dictionary
        with open(corpus_txt_fpath, 'r') as f:
            tot_tokens = 0
            for i,line in enumerate(f):
                words = line.split() + ['<eos>']
                # words = list(map(lambda word_token: VocabUtils.process_word_token({'surface_form':word_token}), words))
                # preprocessing is actually unnecessary since FastText has vectors for @-@, @.@ etc.
                tot_tokens += len(words)
                for word in words:
                    try:
                        prev_freq = vocab_dict[word]
                        vocab_dict[word] = prev_freq + 1
                    except KeyError:
                        vocab_dict[word] = 1
                if i - 1 % lines_logpoint == 0:
                    logging.info("Reading in text corpus to create vocabulary. Line n. "+ str(i+1))

            logging.info("Vocabulary created from " + str(corpus_txt_fpath) + " after processing " + str(tot_tokens) + ' tokens')
    return vocab_dict

        # for i, line in enumerate(open(corpus_txt_fpath, "r", encoding="utf-8")):
        #     if line == '':
        #         break
        #     if i-1 % lines_logpoint == 0:
        #         logging.info("Reading in text corpus to create vocabulary. Line n. "+ str(i+1))
        #     tokens_in_line_truecase, tot_tokens = VocabUtils.process_line(line, tot_tokens)
        #
        #     different_tokens = set(token for token in tokens_in_line_truecase)
        #
        #     update_lts = [(token.replace('_', ' '), line.count(token) ) for token in different_tokens] # '_' was used for phrases.
        #     for word, freq in update_lts:
        #         try:
        #             prev_freq = vocab_dict[word]
        #             vocab_dict[word] = prev_freq + freq
        #         except KeyError:
        #             vocab_dict[word] = freq


    # return vocab_dict


def build_vocabulary_from_senselabeled(lowercase=False):
    vocab_dict = {}

    tokens_toexclude = [Utils.EOS_TOKEN] # + list(string.punctuation)
    # Commas and punctuation signs are present in the Sense-Labeled Corpus as separate tokens.
    # Therefore, it makes sense to keep them in the vocabulary, and thus in the graph, as globals
    slc_split_names = [Utils.TRAINING, Utils.VALIDATION, Utils.TEST]
    for slc_split_name in slc_split_names:
        for token_dict in SLC.read_split(slc_split_name):
            token = VocabUtils.process_word_token(token_dict)
            token = token.lower() if lowercase else token
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
def get_vocabulary_df(senselabeled_or_text, corpus_txt_fpaths, out_vocabulary_h5_filepath, min_count=1, lowercase=False):
    if os.path.exists(out_vocabulary_h5_filepath):
        vocab_df = pd.read_hdf(out_vocabulary_h5_filepath, mode='r')
        logging.info("*** The vocabulary was loaded from the file " + out_vocabulary_h5_filepath)
    else:
        logging.info("*** Creating vocabulary at " + out_vocabulary_h5_filepath)
        vocabulary_h5 = pd.HDFStore(out_vocabulary_h5_filepath, mode='w')
        vocab_h5_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'frequency': Utils.HDF5_BASE_SIZE_512 / 8}

        if senselabeled_or_text:
            vocabulary = build_vocabulary_from_senselabeled(lowercase)
        else:
            vocabulary = build_vocabulary_from_text(corpus_txt_fpaths)

        if min_count>1:
            VocabUtils.eliminate_rare_words(vocabulary, min_count)

        vocab_df = pd.DataFrame(data=zip(vocabulary.keys(), vocabulary.values()), columns=['word', 'frequency'])
        vocabulary_h5.append(key='vocabulary', value=vocab_df, min_itemsize=vocab_h5_itemsizes)
        vocabulary_h5.close()
        logging.info("*** The vocabulary was created. Number of words= " + str(len(vocab_df))+"***")

    return vocab_df