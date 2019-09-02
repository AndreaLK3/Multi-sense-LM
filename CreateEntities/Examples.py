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
import CreateEntities.Vocabulary as Vocabulary

CHUNKSIZE_HDF5 = 128
BATCH_SIZE = 8
WINDOW_RADIUS = 5
D = 300

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

    if os.path.exists(out_hdf5_filepath):
        logging.info("*** The input word pairs had already been created in the file:" + out_hdf5_filepath)
    else:
        logging.info("*** Creating input word pairs from the corpus of examples...")
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


def word_to_vocab_index(word, vocabulary_wordList):

    try:
        return vocabulary_wordList.index(word)
    except ValueError:
        return vocabulary_wordList.index(Utils.UNK_TOKEN)


def main():
    Utils.init_logging(os.path.join("CreateEntities", "Examples.log"), logging.INFO)

    # Load vectors directly from the file
    #pretrained_model_wv = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(Utils.FOLDER_WORD_EMBEDDINGS, Utils.WORD2VEC_FILENAME), binary=True)

    #The vectors and vocabulary can be found (and zipped, if necessary) at: model_wv.vectors, model_wv.index2word

    #pretrained_vocab_size = len(pretrained_model_wv.vectors)  # Number of unique words in our corpus of text ( Vocabulary )

    target_words = ["wide"] # retrieve the updated vectors for those at the end. Will coincide with the vocabulary eventually

    #reduced_vocab = trim_vocabulary_on_corpus_frequency(os.path.join(Utils.FOLDER_WT103, Utils.WT103_TRAIN_FILE))

    # Temporary vocabulary was from: nltk.corpus.words.words()
    extended_lang_id = 'english'
    min_count = 5
    vocabulary_storage_fpath = os.path.join(Utils.FOLDER_WORD_EMBEDDINGS, Utils.WT_MYVOCAB_MINITEST_FILE) # Utils.WT_MYVOCAB_FILE
    vocabulary_source_corpus_fpath = os.path.join(Utils.FOLDER_WT103, Utils.WT_VALID_FILE) # Utils.WT_TRAIN_FILE
    vocabulary_df = Vocabulary.get_vocabulary_df(vocabulary_storage_fpath, vocabulary_source_corpus_fpath, min_count, extended_lang_id)

    # In skip gram architecture of word2vec, the input is the center word and the predictions are the context words.
    # Consider an array of words W, if W(i) is the input (center word), then W(i-2), W(i-1), W(i+1), and W(i+2) are the
    # context words, if the sliding window size is 2.

    ####### Common
    batch_size = BATCH_SIZE
    window_radius = WINDOW_RADIUS

    d = D # len(pretrained_model_wv.vectors[0])   # Number of neurons in the hidden layer of neural network
    #Temporary vocabulary from: nltk
    vocab_size = len(vocabulary_df)
    logging.info('*** Settings: batch_size= ' + str(batch_size) +', window_radius=' + str(window_radius)
                 + ', d=' + str(d) + ', vocab_size=' + str(vocab_size) )
    vocabulary_words_ls = vocabulary_df['word'].to_list()

    ####### Boot-strap version: : No pre-initialization. Skip-Gram over the corpus of examples, then select w ‘s vector

    examples_tokenized_lls = load_input_examples()
    logging.info('*** The input examples from dictionary sources have been loaded.')

    word_pairs_hdf5_filepath = os.path.join(Utils.FOLDER_WORD_EMBEDDINGS, Utils.SKIPGRAM_INPUTWORDPAIRS_FILENAME)
    prepare_input(examples_tokenized_lls, window_radius, word_pairs_hdf5_filepath)
    logging.info("*** Input pairs (centerWord, wordToPredict)")

    inputpairs_hdf5 = pd.read_hdf(word_pairs_hdf5_filepath, mode='r', chunksize=batch_size, iterator = True)
    inputhdf5_df_iterator = inputpairs_hdf5.__iter__()
    batch_gen = SkipGram.BatchGenerator(inputhdf5_df_iterator)
    logging.info('*** Input batch generator')
    #max_iterations = len(word_centerPred_pairs) // batch_size  # in 1 epoch, you can not have more iterations than batches
    random_start_embeddings = np.random.standard_normal((vocab_size,d)).astype(dtype=np.float32)

    inputs_pl, labels_pl, loss = SkipGram.graph(vocab_size, d, batch_size)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    train_loss_summary = tf.summary.scalar('Training_loss', loss)
    init_op = tf.global_variables_initializer()
    logging.info('*** Tensorflow graph ready. Starting Skip-gram training')

    with tf.Session() as sess:
        sess.run(init_op)
        writer_1 = tf.summary.FileWriter(os.path.join("CreateEntities", Utils.SUBFOLDER_TENSORBOARD, train_loss_summary.name), sess.graph)

        while True:
            try:
                batch_input_txt, batch_labels_txt = batch_gen.__next__()
                batch_input = list(map(lambda w: word_to_vocab_index(w, vocabulary_words_ls), batch_input_txt))
                batch_labels = list(map(lambda w: word_to_vocab_index(w, vocabulary_words_ls), batch_labels_txt))

                feed_dict = {inputs_pl: batch_input, labels_pl: batch_labels}
                sess.run([optimizer, writer_1], feed_dict=feed_dict)
            except StopIteration:
                pass


