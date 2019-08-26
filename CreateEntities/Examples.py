import gensim.models
import Utils
import os
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import nltk

CHUNKSIZE_HDF5 = 128


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

def prepare_input(sentences_tokenized_lls, window_radius):
    word_pairs_ls = []

    for sentence_tokens_ls in sentences_tokenized_lls:
        length = len(sentence_tokens_ls)
        for i in range(length):
            center_word = sentence_tokens_ls[i]
            window_words = sentence_tokens_ls[max(0,i-window_radius):i] + sentence_tokens_ls[i+1:i+window_radius+1]
            for w in window_words:
                word_pairs_ls.append((center_word,w))
    return word_pairs_ls

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


class SkipGram_BatchGenerator():

    def __init__(self, inputpairs_indices_ls, batch_size):
        self.inputpairs_indices_ls = inputpairs_indices_ls
        self.batch_size = batch_size
        self.current_pair = 0
        self.num_pairs = len(inputpairs_indices_ls)

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        if self.current_pair > self.num_pairs:
            raise StopIteration
        else:
            self.batch = self.inputpairs_indices_ls[self.current_pair: self.current_pair + self.batch_size]
            self.current_pair = self.current_pair + self.batch_size
            self.inputs = list(map(lambda tpl: tpl[0], self.batch))
            self.labels = list(map(lambda tpl: tpl[1], self.batch))
            return self.inputs, self.labels


def SkipGram_graph(embeddings_atstart, vocab_size, hidden_size_d, batch_size):
    nn_graph = tf.Graph()
    with nn_graph.as_default():
        with tf.variable_scope("SkipGram", reuse=tf.AUTO_REUSE):
            inputs = tf.placeholder(tf.int32, shape=[batch_size])
            labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            W_E = tf.get_variable(name="embeddings", initializer=embeddings_atstart)
            # e.g: model_wv.vectors, or our embeddings, or a random
            # n: shape=(vocab_size, hidden_size_d) -> ValueError: If initializer is a constant, do not specify shape.

            embed = tf.nn.embedding_lookup(W_E, inputs)

            # hidden_layer = tf.matmul(center_word, W_E)

            W_h_out = tf.get_variable(name="weights_h_out", shape=(hidden_size_d, vocab_size),
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases_out = tf.get_variable(name='biases_out', shape=vocab_size, initializer=tf.zeros([vocab_size]))

            # Calculate the loss using negative sampling
            loss = tf.nn.sampled_softmax_loss(W_h_out, biases_out, labels, embed, num_sampled=10, num_classes=vocab_size)
            # tvars = tf.trainable_variables()
            # vars_to_change = [var for var in tvars if not('fixed' in var.name)]

            return inputs, labels, loss


def main():
    Utils.init_logging(os.path.join("CreateEntities", "Examples.log"), logging.INFO)

    # Load vectors directly from the file
    pretrained_model_wv = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(Utils.FOLDER_WORD_EMBEDDINGS, Utils.WORD2VEC_FILENAME), binary=True)
    #The vectors and vocabulary can be found (and zipped, if necessary) at: model_wv.vectors, model_wv.index2word
    vocab_size = len(pretrained_model_wv.vectors)  # Number of unique words in our corpus of text ( Vocabulary )
    d = len(pretrained_model_wv.vectors[0])  # 300   # Number of neurons in the hidden layer of neural network

    target_words = ["wide"] # retrieve the updated vectors for those at the end.


    # In skip gram architecture of word2vec, the input is the center word and the predictions are the context words.
    # Consider an array of words W, if W(i) is the input (center word), then W(i-2), W(i-1), W(i+1), and W(i+2) are the
    # context words, if the sliding window size is 2.

    batch_size = 8


    window_radius = 5
    sentences_tokenized_lls = load_input_examples()
    word_centerPred_pairs = prepare_input(sentences_tokenized_lls, window_radius)

    batch_gen = SkipGram_BatchGenerator(word_centerPred_pairs, batch_size)
    max_iterations = len(word_centerPred_pairs) // batch_size  # in 1 epoch, you can not have more iterations than batches

    inputs_pl, labels_pl, loss = SkipGram_graph(pretrained_model_wv.vectors, vocab_size, d, batch_size)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    train_loss_summary = tf.summary.scalar('Training_loss_Softmax_withNegSampling', loss)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)

        for j in range(0,max_iterations):

            batch_input, batch_labels = batch_gen.__next__()
            feed_dict = {inputs_pl: batch_input, labels_pl: batch_labels}
            sess.run([optimizer, train_loss_summary], feed_dict=feed_dict)


