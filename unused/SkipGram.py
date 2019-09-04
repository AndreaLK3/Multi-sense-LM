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
    train_inputs = tf.placeholder(tf.int32, name="inputs", shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, name="labels", shape=[batch_size, 1])

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