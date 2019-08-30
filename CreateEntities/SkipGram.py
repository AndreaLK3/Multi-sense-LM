import numpy as np
import tensorflow as tf
import pandas as pd


class BatchGenerator():

    def __init__(self, inputpairs_hdf5_df_iterator):
        self.inputpairs_df_iter = inputpairs_hdf5_df_iterator # its chunksize is the batch size

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        self.batch = self.inputpairs_df_iter.__next__()
        self.inputs = list(self.batch['center_word'])
        self.labels = list(self.batch['word_to_predict'])

        return self.inputs, self.labels


def graph(vocabulary_size, hidden_size_d, batch_size):
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



#### Not used:
# def SkipGram_graph_OLD(embeddings_atstart, vocab_size, hidden_size_d, batch_size):
#     nn_graph = tf.Graph()
#     with nn_graph.as_default():
#         with tf.variable_scope("SkipGram", reuse=tf.AUTO_REUSE):
#             inputs = tf.placeholder(tf.int32, shape=[batch_size])
#             labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
#
#             W_E = tf.Variable(embeddings_atstart, name="embeddings")
#
#             # e.g: model_wv.vectors, or our embeddings, or a random
#             # n: shape=(vocab_size, hidden_size_d) -> ValueError: If initializer is a constant, do not specify shape.
#
#             embed = tf.nn.embedding_lookup(W_E, inputs)
#             # hidden_layer = tf.matmul(center_word, W_E)
#
#             W_h_out = tf.get_variable(name="weights_h_out", initializer=tf.random_uniform([vocab_size, hidden_size_d], -1.0, 1.0))
#             biases_out = tf.Variable(tf.zeros(shape=[vocab_size]), name='biases_out')
#
#             # Calculate the loss
#             losses = tf.nn.sampled_softmax_loss(W_h_out, biases_out, labels, embed, num_sampled=5, num_classes=vocab_size)
#             loss = tf.reduce_mean(losses)
#             # tvars = tf.trainable_variables()
#             # vars_to_change = [var for var in tvars if not('fixed' in var.name)]
#
#             return inputs, labels, loss
