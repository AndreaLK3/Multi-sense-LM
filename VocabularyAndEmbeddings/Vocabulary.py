import Filesystem
import Utils
import os
import pandas as pd
import SenseLabeledCorpus as SLC
import logging
import VocabularyAndEmbeddings.Vocabulary_Utilities as VocabUtils
import GetKBInputData.LemmatizeNyms as LN
import nltk
import Filesystem as F

# Auxiliary function: adds a token to the vocabulary (and the lemmatized form too, if needed)
def register_token(token, lemmatizer, vocab_dict):
    try:
        (prev_freq, lemma) = vocab_dict[token]
        vocab_dict[token] = (prev_freq + 1, lemma)
    except KeyError:
        lemmatized_token = LN.lemmatize_term(token, lemmatizer)
        vocab_dict[token] = (1, lemmatized_token)

        if lemmatized_token != token: # also adding to the vocabulary / counting the lemmatized token. 'spring'(s), etc.
            try:
                (prev_freq, lemma) = vocab_dict[lemmatized_token]
                vocab_dict[lemmatized_token] = (prev_freq + 1, lemma)
            except KeyError:
                logging.debug(
                    "Adding lemmatized word '" + lemmatized_token + "' in addition to '" + token + "'")
                vocab_dict[lemmatized_token] = (1, lemmatized_token)
                # the lemmatized form of a lemmatized form is itself

# Read a standard, non sense-labeled text corpus from a file (e.g. WT-2, WT-103)
def build_vocabulary_dict_fromtext(corpus_txt_fpath, vocab_dict, lowercase):
    if vocab_dict is None: vocab_dict = {}

    Utils.init_logging("build_vocabulary_dict_fromtext.log")
    lines_logpoint = 1000
    lemmatizer = nltk.stem.WordNetLemmatizer()

    logging.info(corpus_txt_fpath)
    assert os.path.exists(corpus_txt_fpath)

    with open(corpus_txt_fpath, 'r', encoding="utf-8") as f:
        tot_tokens = 0
        for i,line in enumerate(f):
            # words = line.split() + ['<eos>']
            # FastText has vectors for '@-@', '@.@', but T-XL does not, so we convert them into '-', '.' and join
            tokens, tot_tokens = VocabUtils.process_line(line, tot_tokens)
            tokens = list(map(lambda word_token: VocabUtils.process_word_token({'surface_form':word_token}, lowercasing=lowercase), tokens))
            if len(tokens)>0: logging.debug("line tokens=" + str(tokens))
            for word in tokens:
                register_token(word, lemmatizer, vocab_dict)

            if i - 1 % lines_logpoint == 0:
                logging.info("Reading in text corpus to create vocabulary. Line n. "+ str(i+1))
                logging.info(str(tot_tokens) + ' tokens processed...')
    return vocab_dict


# Read a sense-labeled corpus from a .xml file in the UFSAC notation
def build_vocabulary_dict_from_senselabeled(slc_dir_fpath, vocab_dict, lowercase):

    tokens_toexclude = [] # [Utils.EOS_TOKEN] # + list(string.punctuation)
    # Commas and punctuation signs are present in the Sense-Labeled Corpus as separate tokens.
    # Therefore, it makes sense to keep them in the vocabulary, and thus in the graph, as globals
    lemmatizer = nltk.stem.WordNetLemmatizer()

    for token_dict in SLC.read_split(slc_dir_fpath):
        token = VocabUtils.process_word_token(token_dict, lowercase)
        if token not in tokens_toexclude:
            register_token(token, lemmatizer, vocab_dict)
    if Utils.UNK_TOKEN not in vocab_dict.keys():
        logging.info("Adding manually UNK_TOKEN=" + str(Utils.UNK_TOKEN))
        vocab_dict[Utils.UNK_TOKEN] = (100,Utils.UNK_TOKEN) # add it manually

    return vocab_dict

# Auxiliary function: reunite 2 vocabulary dictionaries, that were created on 2 different corpora
def reunite_vocab_dicts(vocab_dict_1, vocab_dict_2):
    v1_keys_ls = list(vocab_dict_1.keys())
    v2_keys_ls = list(vocab_dict_2.keys())
    all_words = list(set(v1_keys_ls + v2_keys_ls)) # concat lists and apply set, to eliminate duplicates
    reunited_vocab_dict = {}
    for w in all_words:
        freq_1 = 0
        freq_2 = 0
        if w in v1_keys_ls: freq_1 = vocab_dict_1[w][0]
        if w in v2_keys_ls: freq_2 = vocab_dict_2[w][0]
        tot_freq = freq_1 + freq_2
        # the lemmatized form, if present in both, should be identical
        if w in v1_keys_ls: lemma = vocab_dict_1[w][1]
        else: lemma = vocab_dict_2[w][1]
        reunited_vocab_dict[w] = (tot_freq, lemma)

    return reunited_vocab_dict


# Entry function: specify which corpora we wish to create the vocabulary from, and load it or create it as needed
def get_vocabulary_df(corpora_names, lowercase, slc_min_count=2, txt_min_count=1):
    # 1) preparation: selecting the corpora that are to be included
    standardtext_corpus_fpaths = []
    senselabeled_dir_fpaths = []

    if Filesystem.WT2 in corpora_names:
        standardtext_corpus_fpaths.append(os.path.join(F.CORPORA_LOCATIONS[Filesystem.WT2], F.WT_TRAIN_FILE))
    if Filesystem.WT103 in corpora_names:
        standardtext_corpus_fpaths.append(os.path.join(F.CORPORA_LOCATIONS[Filesystem.WT103], F.WT_TRAIN_FILE))
    if Filesystem.SEMCOR in corpora_names:
        senselabeled_dir_fpaths.append(os.path.join(F.CORPORA_LOCATIONS[Filesystem.SEMCOR], F.FOLDER_TRAIN))

    # 2) specifying the filename of the vocabulary. It depends on which corpora were included
    vocab_filename = "vocabulary_" + "_".join(corpora_names) + ".h5"
    vocab_filepath = os.path.join(F.FOLDER_VOCABULARY, vocab_filename)
    logging.info("vocab_filepath=" + str(vocab_filepath))

    # 3) Does the vocabulary file exist already? If yes, just load it
    if os.path.exists(vocab_filepath):
        vocab_df = pd.read_hdf(vocab_filepath, mode='r')
        logging.info("*** The vocabulary was loaded from the file " + vocab_filepath)
        return vocab_df
    else:
        # 4) If the vocabulary did not already exist, it is necessary to create it. Read each corpus one after the other
        logging.info("*** Creating vocabulary at " + vocab_filepath)

        slc_vocab_dict = {}
        txt_vocab_dict = {}

        logging.info("senselabeled_dir_fpaths=" + str(senselabeled_dir_fpaths))
        logging.info("standardtext_corpus_fpaths=" + str(standardtext_corpus_fpaths))

        for slc_dir_fpath in senselabeled_dir_fpaths:
            build_vocabulary_dict_from_senselabeled(slc_dir_fpath, slc_vocab_dict, lowercase)
            if slc_min_count > 1:  # 4b) Removing words with frequency < min_count, if needed
                VocabUtils.eliminate_rare_words(slc_vocab_dict, slc_min_count)
        for txt_corpus_fpath in standardtext_corpus_fpaths:
            build_vocabulary_dict_fromtext(txt_corpus_fpath, txt_vocab_dict, lowercase)
            if txt_min_count > 1:  # 4b) Removing words with frequency < min_count, if needed
                VocabUtils.eliminate_rare_words(txt_vocab_dict, txt_min_count)

        all_vocab_dict = reunite_vocab_dicts(slc_vocab_dict, txt_vocab_dict)

        # 5) Moving the dictionary onto a dataframe
        frequencies = list(map(lambda tpl: tpl[0], all_vocab_dict.values()))
        lemma_forms = list(map(lambda tpl: tpl[1], all_vocab_dict.values()))

        vocab_df = pd.DataFrame(data=zip(all_vocab_dict.keys(), frequencies, lemma_forms), columns=['word', 'frequency', 'lemmatized_form'])
        logging.info("*** The vocabulary was created. Number of words= " + str(len(vocab_df)) + "***")

        vocab_wordls = vocab_df['word'].to_list().copy()
        vocab_df['num_senses'] = [-1]* len(vocab_wordls) # this will be filled up later, when we have both WordNet data and the corpus

        # 6) Writing to HDF5 archive, and to .txt file
        vocab_h5 = pd.HDFStore(vocab_filepath, mode='w')
        vocab_h5_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'frequency': Utils.HDF5_BASE_SIZE_512 / 8,
                              'lemmatized_form': Utils.HDF5_BASE_SIZE_512 / 4,
                              'num_senses': Utils.HDF5_BASE_SIZE_512 / 8}
        vocab_h5.append(key='vocabulary', value=vocab_df, min_itemsize=vocab_h5_itemsizes)
        vocab_h5.close()
        words_ls = list(vocab_df["word"])
        with open(os.path.join(F.FOLDER_VOCABULARY, vocab_filename.replace(".h5", ".txt")), "w") as vocab_txtfile:
            for word in words_ls:
                vocab_txtfile.write(word + "\n")

        return vocab_df