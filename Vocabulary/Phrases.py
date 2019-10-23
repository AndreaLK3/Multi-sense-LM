import nltk
import logging
import gensim
import os

import Filesystem
import Utils
import pandas as pd
import math


# Examine the whole corpus, and create the phrases model,
# that stores a vocabulary that includes a list of phrases (bigrams, e.g. New York).
def create_phrases_model(corpus_txt_filepath):
    all_docsentences = []
    total_num_tokens = 0
    for i, line in enumerate(open(corpus_txt_filepath, "r", encoding="utf-8")):
        line_words = nltk.tokenize.word_tokenize(line)
        total_num_tokens = total_num_tokens + len(line_words)
        all_docsentences.append(line_words)
        if i%10000 == 0 and i !=0:
            logging.info("Phrases: reading in corpus... line n." + str(i))

    min_freq = 50 # math.pow(total_num_tokens, 1/3) // 2
    phrases_score_threshold = 120 # min_freq * 2
    logging.info("Number of tokens in corpus=" + str(total_num_tokens) +
                 " \tParameters for Phrases: min_count=" + str(min_freq) +
                 " \tthreshold=" + str(phrases_score_threshold))
    logging.info("Phrases: Creating the model...")
    phrases_model = gensim.models.phrases.Phrases(sentences=all_docsentences, # needs an lls input on words
                                                  min_count=min_freq, threshold=phrases_score_threshold,
                                                  delimiter=b'_') # Default: min_count=5, threshold=10.0
    logging.info("Phrases: Saving the model...")
    phrases_model.save(os.path.join(Filesystem.FOLDER_VOCABULARY, Filesystem.PHRASES_MODEL_FILE))

    phrases_found_df = pd.DataFrame(set(phrases_model.export_phrases(all_docsentences)))
    phrases_found_df.to_csv(os.path.join(Filesystem.FOLDER_VOCABULARY, "Check_PhrasesFound.csv"))

    del phrases_model


# Then, re-write the phrases-augmented corpus in a new file.
def augment_corpus(in_corpus_txt_filepath, out_corpus_txt_filepath):

    phrases_model = gensim.models.phrases.Phrases.load(os.path.join(Filesystem.FOLDER_VOCABULARY,
                                                                    Filesystem.PHRASES_MODEL_FILE))

    with open(out_corpus_txt_filepath, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(open(in_corpus_txt_filepath, "r", encoding="utf-8")):
            line_withphrases = phrases_model[line.split()]
            outfile.write(' '.join(line_withphrases))
            if i % 10000 == 0 and i != 0:
                logging.info("Phrases: modifying corpus... line n." + str(i))


# Entry point function: if a phrase-processed training corpus has been already created, load it.
# Otherwise, you have to read the original training set and create it.
def setup_phrased_corpus(initial_corpus_fpath, phrased_corpus_fpath):
    if os.path.exists(phrased_corpus_fpath):
        logging.info("*** The training corpus was already put through phrases-processing. Found at " + phrased_corpus_fpath)
        return
    else:
        logging.info(
            "*** Phrases have not been added to the training corpus. Must create Phrases model, etc.")
        create_phrases_model(initial_corpus_fpath)
        augment_corpus(initial_corpus_fpath, phrased_corpus_fpath)
        logging.info("*** Phrases have been added to the training corpus.")