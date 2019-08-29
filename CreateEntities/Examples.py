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


def SkipGram_graph_OLD(embeddings_atstart, vocab_size, hidden_size_d, batch_size):
    nn_graph = tf.Graph()
    with nn_graph.as_default():
        with tf.variable_scope("SkipGram", reuse=tf.AUTO_REUSE):
            inputs = tf.placeholder(tf.int32, shape=[batch_size])
            labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            W_E = tf.Variable(embeddings_atstart, name="embeddings")

            # e.g: model_wv.vectors, or our embeddings, or a random
            # n: shape=(vocab_size, hidden_size_d) -> ValueError: If initializer is a constant, do not specify shape.

            embed = tf.nn.embedding_lookup(W_E, inputs)
            # hidden_layer = tf.matmul(center_word, W_E)

            W_h_out = tf.get_variable(name="weights_h_out", initializer=tf.random_uniform([vocab_size, hidden_size_d], -1.0, 1.0))
            biases_out = tf.Variable(tf.zeros(shape=[vocab_size]), name='biases_out')

            # Calculate the loss
            losses = tf.nn.sampled_softmax_loss(W_h_out, biases_out, labels, embed, num_sampled=5, num_classes=vocab_size)
            loss = tf.reduce_mean(losses)
            # tvars = tf.trainable_variables()
            # vars_to_change = [var for var in tvars if not('fixed' in var.name)]

            return inputs, labels, loss



class CorpusTokenizerIterator():
    def __init__(self, corpus_filepath, batch_lines):
        self.corpus_filepath = corpus_filepath
        self.batch_lines = batch_lines
        self.current_tokens = []
        self.puncts_nohyphen_pattern_str = '[' + string.punctuation.replace('-', '') + ']'
        self.file_handler = open(corpus_filepath, "r")
        self.flag_eof_reached = False

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        if self.flag_eof_reached == True:
            raise StopIteration
        else:
            for i in range(0,self.batch_lines):
                self.next_line = self.file_handler.readline()
                if self.next_line == '':
                    self.flag_eof_reached = True
                self.next_line_noPuncts = re.sub(self.puncts_nohyphen_pattern_str, ' ', self.next_line)
                self.current_tokens.append(nltk.tokenize.word_tokenize(self.next_line_noPuncts))
            return self.current_tokens



def SkipGram_graph(vocabulary_size, hidden_size_d, batch_size):
    # Placeholders for inputs
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    W_E = tf.Variable(tf.random_uniform([vocabulary_size, hidden_size_d], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(W_E, train_inputs)

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, hidden_size_d],
                            stddev=1.0 / np.sqrt(hidden_size_d)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=5,
                       num_classes=vocabulary_size))

    return train_inputs, train_labels, loss




def trim_vocabulary_on_corpus_frequency(corpus_txt_filepath):

    batch_lines = 1024
    corpus_tok_iter = CorpusTokenizerIterator(corpus_txt_filepath, batch_lines)
    vocab_model = gensim.models.Word2Vec()
    vocab_model.build_vocab(corpus_tok_iter.__next__(), update=False);
    while True:
        try:
            corpus_segment = corpus_tok_iter.__next__()
            logging.info('Tokenizing corpus, to create the vocabulary. Processing batch ...' + logging.info(corpus_segment[0]))
            vocab_model.build_vocab(corpus_segment, update=True);
        except StopIteration:
            pass




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
    word_centerPred_pairs = prepare_input(examples_tokenized_lls, window_radius)

    batch_gen = SkipGram_BatchGenerator(word_centerPred_pairs, batch_size)
    max_iterations = len(word_centerPred_pairs) // batch_size  # in 1 epoch, you can not have more iterations than batches
    random_start_embeddings = np.random.standard_normal((vocab_size,d)).astype(dtype=np.float32)

    inputs_pl, labels_pl, loss = SkipGram_graph(vocab_size, d, batch_size)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    train_loss_summary = tf.summary.scalar('Training_loss_Softmax_withNegSampling', loss)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for j in range(0,max_iterations):

            batch_input_txt, batch_labels_txt = batch_gen.__next__()
            batch_input = list(map(lambda w: word_to_vocab_index(w,vocabulary), batch_input_txt))
            batch_labels = list(map(lambda w: word_to_vocab_index(w, vocabulary), batch_labels_txt))

            feed_dict = {inputs_pl: batch_input, labels_pl: batch_labels}
            sess.run([optimizer], feed_dict=feed_dict)


def word_to_vocab_index(word, vocabulary_ls):

    try:
        return vocabulary_ls.index(word)
    except ValueError:
        return vocabulary_ls.index(Utils.UNK_TOKEN)