import gensim.models
import Utils
import os
import logging
import tensorflow as tf
import numpy as np


def main():
    Utils.init_logging(os.path.join("CreateEntities", "Examples.log"), logging.INFO)

    # Load vectors directly from the file
    model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(Utils.FOLDER_WORD_EMBEDDINGS, Utils.WORD2VEC_FILENAME), binary=True)
    # Access vectors for specific words with a keyed lookup:
    vector = model['easy']
    # see the shape of the vector (300,)
    logging.info(vector.shape)
    logging.info(vector[0:10])



    # In skip gram architecture of word2vec, the input is the center word and the predictions are the context words.
    # Consider an array of words W, if W(i) is the input (center word), then W(i-2), W(i-1), W(i+1), and W(i+2) are the
    # context words, if the sliding window size is 2.

    # Defining some variables :

    V = 0 # Number of unique words in our corpus of text ( Vocabulary )
    x = 0  # Input layer (One hot encoding of our input word ).
    d = 300   # Number of neurons in the hidden layer of neural network
    #W = V * d   # Weights between input layer and hidden layer (V x N)
    #W_prime = d * V  # Weights between hidden layer and output layer (N x V)
    y =0   # A softmax output layer having probabilities of every word in our vocabulary


    # If I am using the pretrained embeddings... I should use the weight matrices from the loaded model,
    # and then concatenate to the vocabulary our example entity. Only that row will be updated by the skip-gram loss...
    # note: instead of using the entire matrix, I could extract a meaningful vocabulary, e.g. with 27K instead of 3M words

    # Part of it from: https://github.com/vyomshm/Skip-gram-Word2vec/blob/master/Skip-Gram%2Bword2vec.ipynb

    nn_graph = tf.Graph()
    with nn_graph.as_default():
        center_word_idx = tf.placeholder(dtype=tf.int32, name="input_pl") # vocab. index for the 1-hot encoding input
        target_word_idx = tf.placeholder(dtype=tf.int32, name="label_pl")

        W_E = tf.get_variable(name="embeddings_fixed",shape=(V, d), initializer=tf.random_uniform([V,d],-1, 1))
        W_e_target = tf.get_variable(name="embedding_target",shape=(1, d), initializer=tf.random_uniform([V,d],-1, 1))

        embed = tf.nn.embedding_lookup(W_E,[center_word_idx])

        #hidden_layer = tf.matmul(center_word, W_E)

        W_h_out = tf.get_variable(name="weights_h_out", shape=(d, V), initializer=tf.contrib.layers.xavier_initializer())
        biases_out = tf.get_variable(name='biases_out', shape=d,
                                   initializer=tf.contrib.layers.xavier_initializer())  # tf.zeros([hidden1_units])

        logits = tf.matmul(embed, W_h_out) + biases_out
        softmax_b = tf.Variable(tf.zeros(V))

        # Calculate the loss using negative sampling
        loss = tf.nn.sampled_softmax_loss(logits, softmax_b, target_word_idx, embed, num_sampled=5, num_classes=V)
        tvars = tf.trainable_variables()

        vars_to_change = [var for var in tvars if not('fixed' in var.name)]

        optimizer = tf.train.AdamOptimizer().minimize(loss, var_list=vars_to_change)

