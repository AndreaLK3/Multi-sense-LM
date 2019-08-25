import gensim.models
import Utils
import os
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import nltk

CHUNKSIZE_HDF5 = 128

def main():
    Utils.init_logging(os.path.join("CreateEntities", "Examples.log"), logging.INFO)

    # Load vectors directly from the file
    model_wv = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(Utils.FOLDER_WORD_EMBEDDINGS, Utils.WORD2VEC_FILENAME), binary=True)

    # The vectors and vocabulary can be found (and zipped, if necessary) at: model_wv.vectors, model_wv.index2word

    vocab_size = len(model_wv.vectors)  # Number of unique words in our corpus of text ( Vocabulary )
    d = len(model_wv.vectors[0])  # 300   # Number of neurons in the hidden layer of neural network

    target_words = ["wide"] # retrieve the updated vectors for those at the end.
    all_examples_df_chunksIter = pd.read_hdf(os.path.join(Utils.FOLDER_INPUT, Utils.PREP_EXAMPLES + ".h5")
                                  ,iterator=True, chunksize=CHUNKSIZE_HDF5)

    sentences_tokenized_lls = []
    for chunk in all_examples_df_chunksIter:
        chunk_lls = []
        for row in chunk.itertuples():
            chunk_lls.append(nltk.tokenize.word_tokenize(row.examples))
        sentences_tokenized_lls.extend(chunk_lls)


    # Replace the target word's embedding with a random vector
    # matrix_embeddings = model_wv.vectors
    # matrix_embeddings[center_word_embedding_index] = np.random.normal(loc=0, scale=1.0, size=d)

    # Concatenate the example_entity embedding for our center word, randomly initialized. Its index will be |V|+1
    #model_wv.add(entities=["EXAMPLES_"+center_word], weights=np.random.normal(loc=0, scale=1.0, size=d), replace=False)


    # In skip gram architecture of word2vec, the input is the center word and the predictions are the context words.
    # Consider an array of words W, if W(i) is the input (center word), then W(i-2), W(i-1), W(i+1), and W(i+2) are the
    # context words, if the sliding window size is 2.




    # If I am using the pretrained embeddings... I should use the weight matrices from the loaded model,
    # and then concatenate to the vocabulary our example entity. Only that row will be updated by the skip-gram loss...
    # note: instead of using the entire matrix, I could extract a meaningful vocabulary, e.g. with 27K instead of 3M words

    # Part of it from: https://github.com/vyomshm/Skip-gram-Word2vec/blob/master/Skip-Gram%2Bword2vec.ipynb
    # and from: https://tensorflow.org/tutorials/representation/word2vec

    nn_graph = tf.Graph()
    with nn_graph.as_default():
        center_word_idx = tf.placeholder(dtype=tf.int32, name="input_pl") # vocab. index for the 1-hot encoding input
        word_toPredict_idx = tf.placeholder(dtype=tf.int32, name="label_pl")

        W_E = tf.get_variable(name="embeddings",shape=(vocab_size, d), initializer=model_wv.vectors)

        embed = tf.nn.embedding_lookup(W_E,[center_word_idx])

        #hidden_layer = tf.matmul(center_word, W_E)

        W_h_out = tf.get_variable(name="weights_h_out", shape=(d, vocab_size), initializer=tf.contrib.layers.xavier_initializer())
        biases_out = tf.get_variable(name='biases_out', shape=vocab_size,
                                   initializer=tf.zeros([vocab_size]))  #

        # Calculate the loss using negative sampling
        loss = tf.nn.sampled_softmax_loss(W_h_out, biases_out, word_toPredict_idx, embed, num_sampled=10, num_classes=vocab_size)
        # tvars = tf.trainable_variables()
        # vars_to_change = [var for var in tvars if not('fixed' in var.name)]

        optimizer = tf.train.AdamOptimizer().minimize(loss)

