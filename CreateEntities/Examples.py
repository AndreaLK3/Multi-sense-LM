import gensim.models
import Utils
import os
import logging
import tensorflow as tf
import string
import pandas as pd
import nltk
import re
import numpy as np
import CreateEntities.SkipGram as SkipGram
import itertools
import time

CHUNKSIZE_HDF5 = 128

# Step 0):
# Read the HDF5 storage that contains the examples from the dictionary sources, after they were pre-processed.
# Load them in tokenized form (e.g.: ['summer', 'temperatures', 'reached', 'all-time', 'high'])
# in a list-of-lists, currently in memory
def load_input_examples():
    all_examples_df_chunksIter = pd.read_hdf(os.path.join(Utils.FOLDER_INPUT, Utils.PREP_EXAMPLES + ".h5")
                                  ,iterator=True, chunksize=CHUNKSIZE_HDF5)
    sentences_tokenized_lls = []

    for chunk in all_examples_df_chunksIter:
        chunk_lls = []
        for row in chunk.itertuples():
            chunk_lls.append(nltk.tokenize.word_tokenize(row.examples))
        sentences_tokenized_lls.extend(chunk_lls)
    return sentences_tokenized_lls

# Step 1): Create the tuples (center_word, word_to_predict)
# Stored in a HDF5 file
def prepare_input(sentences_tokenized_lls, window_radius, out_hdf5_filepath):

    with pd.HDFStore(out_hdf5_filepath, mode='w') as out_hdf5_file: # reset and open
        df_columns = ['center_word', 'word_to_predict']

        for sentence_tokens_ls in sentences_tokenized_lls:
            length = len(sentence_tokens_ls)
            sentence_word_pairs = []
            for i in range(length):
                center_word = sentence_tokens_ls[i]
                window_words = sentence_tokens_ls[max(0,i-window_radius):i] + sentence_tokens_ls[i+1:i+window_radius+1]
                for w in window_words:
                    sentence_word_pairs.append((center_word,w))
            df = pd.DataFrame(data=sentence_word_pairs, columns=df_columns)
            out_hdf5_file.append(key='skipgram_input', value=df,
                                 min_itemsize={key: Utils.HDF5_BASE_CHARSIZE/2 for key in df_columns})
    return


def input_to_indices(word_pairs_ls, vocabulary_wordlist, oov_index):

    input_indices_ls = []

    for word_pair in word_pairs_ls:
        center_word = word_pair[0]
        word_toPredict = word_pair[1]
        try:
            center_word_index = vocabulary_wordlist.index(center_word)
        except ValueError:
            center_word_index = oov_index
        try:
            word_toPredict_index = vocabulary_wordlist.index(word_toPredict)
        except ValueError:
            word_toPredict_index = oov_index
        input_indices_ls.append((center_word_index, word_toPredict_index))

    return input_indices_ls


class CorpusTokenizerIterator():
    def __init__(self, corpus_filepath, batch_lines):
        self.corpus_filepath = corpus_filepath
        self.batch_lines = batch_lines
        self.puncts_nohyphen_pattern_str = '[' + string.punctuation.replace('-', '') + ']'
        self.file_handler = open(corpus_filepath, "r")
        #self.flag_eof_reached = False

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        self.current_tokens = []
        for i in range(0,self.batch_lines):
            self.next_line = self.file_handler.readline()
            logging.debug('{'+ self.next_line +'}')
            if self.next_line == '':
                raise StopIteration
            self.next_line_noPuncts = re.sub(self.puncts_nohyphen_pattern_str, ' ', self.next_line)
            self.current_tokens.append(nltk.tokenize.word_tokenize(self.next_line_noPuncts))
        return self.current_tokens


def word_to_vocab_index(word, vocabulary_ls):

    try:
        return vocabulary_ls.index(word)
    except ValueError:
        return vocabulary_ls.index(Utils.UNK_TOKEN)




def main():
    Utils.init_logging(os.path.join("CreateEntities", "Examples.log"), logging.INFO)

    # Load vectors directly from the file
    #pretrained_model_wv = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(Utils.FOLDER_WORD_EMBEDDINGS, Utils.WORD2VEC_FILENAME), binary=True)

    #The vectors and vocabulary can be found (and zipped, if necessary) at: model_wv.vectors, model_wv.index2word

    #pretrained_vocab_size = len(pretrained_model_wv.vectors)  # Number of unique words in our corpus of text ( Vocabulary )

    target_words = ["wide"] # retrieve the updated vectors for those at the end. Will coincide with the vocabulary eventually

    #reduced_vocab = trim_vocabulary_on_corpus_frequency(os.path.join(Utils.FOLDER_WT103, Utils.WT103_TRAIN_FILE))
    # Temporary vocabulary from: nltk
    vocabulary = nltk.corpus.words.words()
    vocabulary_h5 = pd.HDFStore(os.path.join(Utils.FOLDER_WORD_EMBEDDINGS, Utils.WT_MYVOCAB_FILE), mode='w')
    vocab_h5_itemsizes = {'word': Utils.HDF5_BASE_CHARSIZE / 4, 'frequency': Utils.HDF5_BASE_CHARSIZE / 8}
    vocab_df = pd.DataFrame(data=list(zip(vocabulary.keys(), vocabulary.values())), columns=['word','frequency'])
    vocabulary_h5.append(key='vocabulary', value=vocab_df, min_itemsize=vocab_h5_itemsizes)

    # In skip gram architecture of word2vec, the input is the center word and the predictions are the context words.
    # Consider an array of words W, if W(i) is the input (center word), then W(i-2), W(i-1), W(i+1), and W(i+2) are the
    # context words, if the sliding window size is 2.


    ####### Common
    batch_size = 8
    window_radius = 5

    d = 300 # len(pretrained_model_wv.vectors[0])   # Number of neurons in the hidden layer of neural network
    #Temporary vocabulary from: nltk
    vocab_size = len(vocabulary)

    ####### Boot-strap version: : No pre-initialization. Skip-Gram over the corpus of examples, then select w ‘s vector

    examples_tokenized_lls = load_input_examples()

    word_pairs_hdf5_filepath = os.path.join(Utils.FOLDER_WORD_EMBEDDINGS, Utils.SKIPGRAM_INPUTWORDPAIRS_FILENAME)
    prepare_input(examples_tokenized_lls, window_radius, word_pairs_hdf5_filepath)
    inputpairs_hdf5 = pd.read_hdf(word_pairs_hdf5_filepath, mode='r', chunksize=batch_size, iterator = True)
    inputhdf5_df_iterator = inputpairs_hdf5.__iter__()

    batch_gen = SkipGram.BatchGenerator(inputhdf5_df_iterator)
    #max_iterations = len(word_centerPred_pairs) // batch_size  # in 1 epoch, you can not have more iterations than batches
    random_start_embeddings = np.random.standard_normal((vocab_size,d)).astype(dtype=np.float32)

    inputs_pl, labels_pl, loss = SkipGram.graph(vocab_size, d, batch_size)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    train_loss_summary = tf.summary.scalar('Training_loss_Softmax_withNegSampling', loss)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for j in range(0,1000000): #max_iterations

            batch_input_txt, batch_labels_txt = batch_gen.__next__()
            batch_input = list(map(lambda w: word_to_vocab_index(w,vocabulary), batch_input_txt))
            batch_labels = list(map(lambda w: word_to_vocab_index(w, vocabulary), batch_labels_txt))

            feed_dict = {inputs_pl: batch_input, labels_pl: batch_labels}
            sess.run([optimizer], feed_dict=feed_dict)








def create_vocabulary_from_corpus(corpus_txt_filepath):

    #corpus_tok_iter = CorpusTokenizerIterator(corpus_txt_filepath, batch_lines)
    vocab_dict = {}
    tot_tokens = 0
    w2v_model = gensim.models.Word2Vec(min_count=1)

    i = 0
    time_prev = time.time()
    for i, line in enumerate(open(corpus_txt_filepath, "r", encoding="utf-8")):
        if line == '':
            break
        # tokens_in_line = nltk.tokenize.word_tokenize(line)
        line_noPuncts = re.sub('[' + string.punctuation.replace('-', '') + ']', ' ', line)
        tokens_in_line = nltk.tokenize.word_tokenize(line_noPuncts)
        tot_tokens = tot_tokens + len(tokens_in_line)

        if i % 2000 == 0:
            time_next = time.time()
            time_elapsed = round( time_next - time_prev, 4)
            print("Reading in line n. : " + str(i) + ' ; number of tokens encountered: ' + str(tot_tokens) +
                  " ; time elapsed = " + str(time_elapsed) + " s")
            time_prev = time.time()

        different_tokens = set(token for token in tokens_in_line)

        update_lts = [(token, line.count(token) ) for token in different_tokens]
        for word, freq in update_lts:
            try:
                prev_freq = vocab_dict[word]
                vocab_dict[word] = prev_freq + freq
            except KeyError:
                vocab_dict[word] = freq

    return vocab_dict
