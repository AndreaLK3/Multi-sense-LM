import WordEmbeddings.EmbedWithDBERT as EDB
import WordEmbeddings.EmbedWithFastText as EFT
import pandas as pd
import os
import Filesystem as F
import logging
import Utils
from enum import Enum
import transformers

class Method_for_SPV(Enum):
    DISTILBERT = 1
    FASTTEXT = 2


# The main function of the module: iterate over the vocabulary that we previously did build from the training corpus,
# and use either DistilBERT or FastText to compute d=768 or d=300 single-prototype word embeddings.
def compute_single_prototype_embeddings(vocabulary_h5_filepath, sp_out_h5_fpath, method):
    Utils.init_logging('compute_single_prototype_embeddings.log')
    vocab_df = pd.read_hdf(vocabulary_h5_filepath, mode='r')
    sp_out_archive = pd.HDFStore(sp_out_h5_fpath, mode='w')

    if method == Method_for_SPV.DISTILBERT:
        distilBERT_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                                    output_hidden_states=True)
        distilBERT_tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    fasttext_vectors = EFT.load_fasttext_vectors()
    num_vectors_dump = 100
    i = 0
    word_vectors_lts = []
    sp_h5_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'vector': Utils.HDF5_BASE_SIZE_512}

    for idx_word_freq_tpl in vocab_df.itertuples():
        word = idx_word_freq_tpl[1]

        if method == Method_for_SPV.DISTILBERT:
            word_vector = EDB.compute_sentence_dBert_vector(distilBERT_model, distilBERT_tokenizer, word)
        else: # i.e. elif method == Method_for_SPV.FASTTEXT:
            word_vector = fasttext_vectors[word]

        word_vectors_lts.append((word, word_vector))
        i = i+1
        if i % num_vectors_dump == 0:
            w_vectors_df = pd.DataFrame(word_vectors_lts, columns=['word', 'vector'])
            sp_out_archive.append(key='vocabulary', value=vocab_df, min_itemsize=sp_h5_itemsizes)
            logging.info(w_vectors_df)
            # logging.info(w_vectors_df.dtypes)
            # pd.to_numeric(w_vectors_df['vector'])
            # w_vectors_df['word'] = w_vectors_df['word'].astype('|S256')
            # logging.info(w_vectors_df.dtypes)
            # w_vectors_df.to_hdf(sp_out_archive, key='SPVs', format='table', append='True')

            # reset
            i = 0
            word_vectors_lts = []

    vocabulary_h5_filepath.close()
    sp_out_h5_fpath.close()

