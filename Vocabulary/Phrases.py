import nltk
import logging
import gensim
import os
import Utils
import pandas as pd

# Examine the whole corpus, and create the phrases model,
# that stores a vocabulary that includes a list of phrases (bigrams, e.g. New York).

def create_phrases_model(corpus_txt_filepath):
    all_docsentences = []
    total_num_tokens = 0
    for i, line in enumerate(open(corpus_txt_filepath, "r", encoding="utf-8")):
        line_words = nltk.tokenize.word_tokenize(line)
        total_num_tokens = total_num_tokens + len(line_words)
        all_docsentences.extend(line)
        if i%10000 == 0 and i !=0:
            logging.info("Phrases: reading in corpus... line n." + str(i))
            logging.info(line)


    min_freq = 40 # total_num_tokens // 50000
    phrases_score_threshold = 150 # min_freq * 5
    logging.info("Number of tokens in corpus=" + str(total_num_tokens) +
                 " \tParameters for Phrases: min_count=" + str(min_freq) +
                 " \tthreshold=" + str(phrases_score_threshold))
    logging.info("Phrases: Creating the model...")
    phrases_model = gensim.models.phrases.Phrases(sentences=all_docsentences,
                                                  min_count=min_freq, threshold=phrases_score_threshold,
                                                  delimiter=b'_') # Default: min_count=5, threshold=10.0
    logging.info("Phrases: Saving the model...")
    phrases_model.save(os.path.join(Utils.FOLDER_VOCABULARY, Utils.PHRASES_MODEL_FILE))

    logging.info(phrases_model.vocab)
    pd.DataFrame.from_dict(phrases_model.vocab)
    raise Exception
    del phrases_model


# Then, re-write the phrases-augmented corpus in a new file.
def augment_corpus(in_corpus_txt_filepath, out_corpus_txt_filepath):

    phrases_model = gensim.models.phrases.Phrases.load(os.path.join(Utils.FOLDER_VOCABULARY, Utils.PHRASES_MODEL_FILE))

    with open(out_corpus_txt_filepath, "r", encoding="utf-8") as outfile:
        for i, line in enumerate(open(in_corpus_txt_filepath, "r", encoding="utf-8")):
            line_withphrases = phrases_model[line]
            outfile.write(line_withphrases)
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