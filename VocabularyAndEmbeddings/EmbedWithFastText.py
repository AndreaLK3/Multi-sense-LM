import os
import Filesystem as F
import io
import logging
import Utils
import nltk
import gensim.models.fasttext
import Filesystem as F

FASTEXT_DIMENSIONS = 300



def get_sentence_avg_vector(sentence_str, fasttext_vectors):
    sentence_tokens = nltk.tokenize.word_tokenize(sentence_str)
    sum_of_vectors = [0] * FASTEXT_DIMENSIONS

    for token in sentence_tokens:
        token_vector = fasttext_vectors[token]
        sum_of_vectors = sum_of_vectors + token_vector

    vectors_avg = sum_of_vectors / len(sentence_tokens)
    return vectors_avg



def load_fasttext_vectors():

    fasttext_fpath = os.path.join(F.FOLDER_VOCABULARY, F.FASTTEXT_PRETRAINED_EMBEDDINGS_FILE)
    fasttext_vectors = gensim.models.fasttext.load_facebook_vectors(fasttext_fpath)

    return fasttext_vectors

