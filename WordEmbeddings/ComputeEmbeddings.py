import sqlite3

import Filesystem
import WordEmbeddings.EmbedWithDBERT as EDB
import WordEmbeddings.EmbedWithFastText as EFT
import pandas as pd
import os
import numpy as np
import Filesystem as F
import logging
import Utils
from enum import Enum
import transformers

from WordEmbeddings.EmbedWithDBERT import compute_sentence_dBert_vector


class Method(Enum):
    DISTILBERT = Utils.DISTILBERT
    FASTTEXT = Utils.FASTTEXT


# The main function of the module: iterate over the vocabulary that we previously did build from the training corpus,
# and use either DistilBERT or FastText to compute d=768 or d=300 single-prototype word embeddings.
def compute_single_prototype_embeddings(vocabulary_df, spvs_out_fpath, method):

    if method == Method.DISTILBERT:
        distilBERT_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                                    output_hidden_states=True)
        distilBERT_tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:  # i.e. elif method == Method_for_SPV.FASTTEXT:
        fasttext_vectors = EFT.load_fasttext_vectors()

    num_vectors_dump = 100
    i = 0
    word_vectors_lls = []

    for idx_word_freq_tpl in vocabulary_df.itertuples():
        word = idx_word_freq_tpl[1]

        if method == Method.DISTILBERT:
            word_vector = EDB.compute_sentence_dBert_vector(distilBERT_model, distilBERT_tokenizer, word).squeeze().numpy()
        else: # i.e. elif method == Method_for_SPV.FASTTEXT:
            word_vector = fasttext_vectors[word]

        word_vectors_lls.append(word_vector)
        i = i+1
        if i % num_vectors_dump == 0:
            embds_nparray = np.array(word_vectors_lls)
            np.save(spvs_out_fpath, embds_nparray)
            # reset
            i = 0
            word_vectors_lls = []
    logging.info('Computed the single-prototype embeddings for the vocabulary tokens, at: ' + spvs_out_fpath)


# In the previous step, we stored the start-and-end indices of elements in a Sqlite3 database.
# It is necessary to write the embeddings in the .npy file with the correct ordering.
def compute_elements_embeddings(elements_name, method):

    if method == Method.DISTILBERT:
        distilBERT_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                                        output_hidden_states=True)
        distilBERT_tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    else:  # i.e. elif method == Method_for_SPV.FASTTEXT:
        fasttext_vectors = EFT.load_fasttext_vectors()

    input_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.PROCESSED + '_' + elements_name + ".h5")
    output_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.VECTORIZED + '_' + str(method.value) + '_'
                                   + elements_name) # + ".npy"
    vocabTable_db_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.INDICES_TABLE_DB)

    input_db = pd.HDFStore(input_filepath, mode='r')
    vocabTable_db = sqlite3.connect(vocabTable_db_filepath)
    vocabTable_db_c = vocabTable_db.cursor()

    matrix_of_sentence_embeddings = []
    vocabTable_db_c.execute("SELECT * FROM vocabulary_table")

    for row in vocabTable_db_c: # consuming the cursor iterator. Tuple returned: ('wide.a.1', 2, 4,5, 16,18)
        sense_df = Utils.select_from_hdf5(input_db, elements_name, [Utils.SENSE_WN_ID], [row[0]])
        element_text_series = sense_df[elements_name]
        for element_text in element_text_series:
            logging.debug("ComputeEmbeddings.compute_elements_embeddings(elements_name, method) > " +
                         " wn_id=row[0]=" + str(row[0]) + " ;  elements_name=" + str(elements_name) +
                         " ; element_text=" + str(element_text))
            if method == Method.DISTILBERT:
                vector = compute_sentence_dBert_vector(distilBERT_model, distilBERT_tokenizer, element_text).squeeze().numpy()
            else: # i.e. elif method == Method_for_SPV.FASTTEXT:
                vector = EFT.get_sentence_avg_vector(element_text, fasttext_vectors)
            matrix_of_sentence_embeddings.append(vector)

    embds_nparray = np.array(matrix_of_sentence_embeddings)
    np.save(output_filepath, embds_nparray)

    logging.info('Computed the embeddings for the dictionary elements: ' + elements_name +
                 " , saved at: " + output_filepath)