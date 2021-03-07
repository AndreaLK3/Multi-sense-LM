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
def register_token(token, lemmatizer, vocab_dict, lemmatized_forms_ls):
    lemmatized_token = LN.lemmatize_term(token, lemmatizer)
    try:
        prev_freq = vocab_dict[token]
        vocab_dict[token] = prev_freq + 1
    except KeyError:
        vocab_dict[token] = 1
        lemmatized_forms_ls.append(lemmatized_token)

    if lemmatized_token != token: # also adding the lemmatized token to the vocabulary. 'spring'(s), etc.
        try:
            prev_freq = vocab_dict[lemmatized_token]
            vocab_dict[lemmatized_token] = prev_freq + 1
        except KeyError:
            logging.info(
                "Adding lemmatized word '" + lemmatized_token + "' in addition to '" + token + "'")
            vocab_dict[lemmatized_token] = 1
            lemmatized_forms_ls.append(lemmatized_token)  # the lemmatized form of a lemmatized form is itself

# Read a standard, non sense-labeled text corpus from a file (e.g. WT-2, WT-103)
def build_vocabulary_dict_fromtext(corpus_txt_fpath, vocab_dict, lemmatized_forms_ls, lowercase):
    if vocab_dict == None: vocab_dict = {}
    if lemmatized_forms_ls == None: lemmatized_forms_ls = []

    Utils.init_logging("build_vocabulary_dict_fromtext.log")
    lines_logpoint = 1000
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # """Tokenizes a text file."""
    assert os.path.exists(corpus_txt_fpath)
    # Add words to the dictionary
    with open(corpus_txt_fpath, 'r', encoding="utf-8") as f:
        tot_tokens = 0
        for i,line in enumerate(f):
            # words = line.split() + ['<eos>']
            # FastText has vectors for '@-@', '@.@', but T-XL does not, so we convert them into '-', '.' and join
            tokens, tot_tokens = VocabUtils.process_line(line, tot_tokens)
            tokens = list(map(lambda word_token: VocabUtils.process_word_token({'surface_form':word_token}, lowercasing=lowercase), tokens))
            if len(tokens)>0: logging.info("line tokens=" + str(tokens))
            for word in tokens:
                register_token(word, lemmatizer, vocab_dict, lemmatized_forms_ls)

            if i - 1 % lines_logpoint == 0:
                logging.info("Reading in text corpus to create vocabulary. Line n. "+ str(i+1))

            logging.info("VocabularyAndEmbeddings created from " + str(corpus_txt_fpath) + " after processing " + str(tot_tokens) + ' tokens')
    return vocab_dict, lemmatized_forms_ls

# Read a sense-labeled corpus from a .xml file in the UFSAC notation
def build_vocabulary_dict_from_senselabeled(slc_corpus_fpath, lowercase, vocab_dict, lemmatized_forms_ls):
    if vocab_dict == None: vocab_dict = {}
    if lemmatized_forms_ls == None: lemmatized_forms_ls = []

    tokens_toexclude = [] # [Utils.EOS_TOKEN] # + list(string.punctuation)
    # Commas and punctuation signs are present in the Sense-Labeled Corpus as separate tokens.
    # Therefore, it makes sense to keep them in the vocabulary, and thus in the graph, as globals
    lemmatizer = nltk.stem.WordNetLemmatizer()

    for token_dict in SLC.read_split(slc_corpus_fpath):
        token = VocabUtils.process_word_token(token_dict, lowercase)
        if token not in tokens_toexclude:
            register_token(token, lemmatizer, vocab_dict, lemmatized_forms_ls)
    if Utils.UNK_TOKEN not in vocab_dict.keys():
        logging.info("Adding manually UNK_TOKEN=" + str(Utils.UNK_TOKEN))
        vocab_dict[Utils.UNK_TOKEN] = 100 # add it manually
        lemmatized_forms_ls.append(Utils.UNK_TOKEN)

    return vocab_dict, lemmatized_forms_ls


# Entry function: specify which corpuses we wish to create the vocabulary from, and load it or create it as needed
def get_vocabulary_df(corpuses, lowercase, slc_min_count, txt_min_count=1):
    # 1) preparation: seelcting the text corpuses that are to be included
    standardtext_corpus_fpaths = []
    senselabeled_corpus_fpaths = []
    if Utils.WT2 in corpuses:
        standardtext_corpus_fpaths.append(os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_STANDARDTEXT, F.FOLDER_WT2,
                                                       "train.txt"))
    if Utils.WT103 in corpuses:
        standardtext_corpus_fpaths.append(os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_STANDARDTEXT, F.FOLDER_WT103,
                                                       "train.txt"))
    if Utils.SEMCOR in corpuses:
        senselabeled_corpus_fpaths.append(os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_SENSELABELED, F.FOLDER_SEMCOR,
                                                       F.FOLDER_TRAIN, "semcor.xml"))

    # 2) specifying the filename of the vocabulary. It depends on which corpuses were included
    vocab_filename = "vocabulary_" + "_".join(corpuses) + ".h5"
    vocab_filepath = os.path.join(F.FOLDER_VOCABULARY, vocab_filename)

    # 3) Does the vocabulary file exist already? If yes, we do not need to read the text to create it
    if os.path.exists(vocab_filepath):
        vocab_df = pd.read_hdf(vocab_filepath, mode='r')
        logging.info("*** The vocabulary was loaded from the file " + vocab_filepath)

    # 4) If the vocabulary did not already exist, it is necessary to create it. Read each corpus one after the other
    logging.info("*** Creating vocabulary at " + vocab_filepath)

    all_vocab_dict ={}
    all_lemmatized_forms_ls = []

    for txt_corpus_fpath in standardtext_corpus_fpaths:
        build_vocabulary_dict_fromtext(txt_corpus_fpath, all_vocab_dict, all_lemmatized_forms_ls, lowercase)
    for slc_corpus_fpath in senselabeled_corpus_fpaths:
        build_vocabulary_dict_from_senselabeled(slc_corpus_fpath, lowercase, all_vocab_dict, all_lemmatized_forms_ls)

    # 5) Reuniting the words, frequency and lemmatized forms in 1 dictionary
    word_keys = list(all_vocab_dict.keys())
    freq_values = list(all_vocab_dict.values())
    new_freq_lemma_values = list(zip(freq_values, all_lemmatized_forms_ls))
    full_vocab_dict = dict(zip(word_keys, new_freq_lemma_values))

    # 6) Removing words with frequency < min_count, if needed
    if min_count > 1:
        VocabUtils.eliminate_rare_words(full_vocab_dict, min_count)

    # 7) Moving the dictionary onto a dataframe, and writing it to HDF5 archive
    frequencies = list(map(lambda tpl: tpl[0], full_vocab_dict.values()))
    lemma_forms = list(map(lambda tpl: tpl[1], full_vocab_dict.values()))

    vocab_df = pd.DataFrame(data=zip(full_vocab_dict.keys(), frequencies, lemma_forms), columns=['word', 'frequency', 'lemmatized_form'])
    logging.info("*** The vocabulary was created. Number of words= " + str(len(vocab_df)) + "***")

    vocab_wordls = vocab_df['word'].to_list().copy()
    vocab_df['num_senses'] = [-1]* len(vocab_wordls) # this will be filled up later, when we have both WordNet data and the corpus

    vocab_h5 = pd.HDFStore(vocab_filepath, mode='w')
    vocab_h5_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'frequency': Utils.HDF5_BASE_SIZE_512 / 8,
                          'lemmatized_form': Utils.HDF5_BASE_SIZE_512 / 4,
                          'num_senses': Utils.HDF5_BASE_SIZE_512 / 8}
    vocab_h5.append(key='vocabulary', value=vocab_df, min_itemsize=vocab_h5_itemsizes)
    vocab_h5.close()

    return vocab_df