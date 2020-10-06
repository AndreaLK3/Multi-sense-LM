import Utils
import os
import pandas as pd
import SenseLabeledCorpus as SLC
import logging
import Vocabulary.Vocabulary_Utilities as VocabUtils
import PrepareKBInput.LemmatizeNyms as LN
import nltk
import Filesystem as F

def build_vocabulary_dict_fromtext(corpus_txt_fpaths):
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


def build_vocabulary_dict_from_senselabeled(lowercase):
    vocab_dict = {}

    tokens_toexclude = [Utils.EOS_TOKEN] # + list(string.punctuation)
    # Commas and punctuation signs are present in the Sense-Labeled Corpus as separate tokens.
    # Therefore, it makes sense to keep them in the vocabulary, and thus in the graph, as globals
    slc_split_names = [Utils.TRAINING, Utils.VALIDATION] # , , Utils.TEST
    lemmatizer = nltk.stem.WordNetLemmatizer()

    for slc_split_name in slc_split_names:
        for token_dict in SLC.read_split(slc_split_name):
            token = VocabUtils.process_word_token(token_dict, lowercase)

            lemmatized_token = LN.lemmatize_term(token, lemmatizer)
            if token not in tokens_toexclude:
                try:
                    prev_freq = vocab_dict[token]
                    vocab_dict[token] = prev_freq + 1
                except KeyError:
                    vocab_dict[token] = 1

                if lemmatized_token != token: # also adding the lemmatized token to the vocabulary. 'spring'(s), etc.
                    try:
                        prev_freq = vocab_dict[lemmatized_token]
                        vocab_dict[lemmatized_token] = prev_freq + 1
                    except KeyError:
                        vocab_dict[lemmatized_token] = 1
        if Utils.UNK_TOKEN not in vocab_dict.keys():
            logging.info("Adding manually UNK_TOKEN=" + str(Utils.UNK_TOKEN))
            vocab_dict[Utils.UNK_TOKEN] = 100 # add it manually

    return vocab_dict


# Entry function: if a vocabulary is already present in the specified path, load it. Otherwise, create it.
def get_vocabulary_df(senselabeled_or_text, vocabulary_h5_filepath, textcorpus_fpaths, min_count, lowercase):

    if os.path.exists(vocabulary_h5_filepath):
        vocab_df = pd.read_hdf(vocabulary_h5_filepath, mode='r')
        logging.info("*** The vocabulary was loaded from the file " + vocabulary_h5_filepath)
    else:
        logging.info("*** Creating vocabulary at " + vocabulary_h5_filepath)
        vocabulary_h5 = pd.HDFStore(vocabulary_h5_filepath, mode='w')
        vocab_h5_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'frequency': Utils.HDF5_BASE_SIZE_512 / 8,
                              'lemmatized_form': Utils.HDF5_BASE_SIZE_512 / 4, 'num_senses': Utils.HDF5_BASE_SIZE_512 / 8}

        if senselabeled_or_text:
            vocabulary_wordfreq_dict = build_vocabulary_dict_from_senselabeled(lowercase)
        else:
            vocabulary_wordfreq_dict = build_vocabulary_dict_fromtext(textcorpus_fpaths)

        if min_count>1:
            VocabUtils.eliminate_rare_words(vocabulary_wordfreq_dict, min_count)

        vocab_df = pd.DataFrame(data=zip(vocabulary_wordfreq_dict.keys(), vocabulary_wordfreq_dict.values()), columns=['word', 'frequency'])
        logging.info("*** The vocabulary was created. Number of words= " + str(len(vocab_df)) + "***")

        vocab_wordls = vocab_df['word'].to_list().copy()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized_forms = [LN.lemmatize_term(word, lemmatizer) for word in vocab_wordls]
        vocab_df['lemmatized_form'] = lemmatized_forms
        vocab_df['num_senses'] = [-1]* len(vocab_wordls) # this will be filled up later, when we have both WordNet data and the corpus

        vocabulary_h5.append(key='vocabulary', value=vocab_df, min_itemsize=vocab_h5_itemsizes)
        vocabulary_h5.close()

    return vocab_df