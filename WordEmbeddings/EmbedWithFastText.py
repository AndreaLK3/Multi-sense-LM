import os
import Filesystem as F
import io
import logging
import Utils
import nltk
import gensim.models.fasttext

FASTEXT_DIMENSIONS = 300

# ##### Made from the already-provided function to load the FastText word vectors stored in text files
# def load_vectors(fname):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for i,line in enumerate(fin):
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#         if i % 10000 == 0:
#             logging.info("Loading FastText vectors from text file... Line n.: " + str(i))
#     return data
# #####


def get_word_vector(word, fasttext_vectors):
    return fasttext_vectors[word]


def get_sentence_avg_vector(sentence_str, fasttext_vectors):
    sentence_tokens = nltk.tokenize.word_tokenize(sentence_str)
    sum_of_vectors = [0] * FASTEXT_DIMENSIONS

    for token in sentence_tokens:
        token_vector = fasttext_vectors[token]
        sum_of_vectors = sum_of_vectors + token_vector

    vectors_avg = sum_of_vectors / len(sentence_tokens)
    return vectors_avg



def test():
    Utils.init_logging(os.path.join(F.FOLDER_WORD_EMBEDDINGS, "EmbedWithFastText.log"))
    fasttext_fpath = os.path.join(F.FOLDER_WORD_EMBEDDINGS, F.FASTTEXT_EMBEDDINGS_FILE)
    fasttext_vectors = gensim.models.fasttext.load_facebook_vectors(fasttext_fpath)

    return fasttext_vectors
