import os
import sqlite3
import numpy as np
import pandas as pd
import transformers
import torch
import logging
import Filesystem
import Utils
from WordEmbeddings.AutoEncoderDBERT import SentenceEmbeddingEncoder

START_SENTENCE_TOKEN = "[CLS]"
END_SEP_TOKEN = "[SEP]"


# Alternative: currently the trained encoder is not in use
def compute_300d_sentence_vector(distilbert_sentence_vector):
    encoder_model = SentenceEmbeddingEncoder()
    encoder_model.load_state_dict(torch.load(os.path.join(Filesystem.FOLDER_WORD_EMBEDDINGS, Filesystem.ENCODER_MODEL)))
    encoder_model.eval()
    sentence_embedding = encoder_model(distilbert_sentence_vector)
    return sentence_embedding


def compute_sentence_dBert_vector(model, tokenizer, sentence_text):

    toks = tokenizer.tokenize(START_SENTENCE_TOKEN + sentence_text + END_SEP_TOKEN)
    indices = tokenizer.convert_tokens_to_ids(toks)

    segment_ids = [1] * len(indices)# single-sentence inputs only require a series of 1s
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(indices).unsqueeze(0)
    segment_tensor = torch.tensor(segment_ids).unsqueeze(0)

    with torch.no_grad():
        last_layer = model(tokens_tensor, segment_tensor)[0]
        # last_hidden_state: torch.FloatTensor of shape (batch_size, sequence_length, hidden_size) (here [1, 17, 768])

    # To get a single vector for our entire sentence we have multiple application-dependent choices, in terms of
    # methods (mean, max, concatenation, etc.) and layers used (last four, all, last layer, etc.).
    # A simple approach is to average the (/second-to-)last hidden layer of each token, producing one 768-length vector

    sentence_embedding = torch.mean(last_layer, dim=1)[0] # batch size 1
    logging.debug(sentence_embedding.shape)
    return sentence_embedding


# In the previous step, we stored the start-and-end indices of elements in a Sqlite3 database.
# It is necessary to write the embeddings in the .npy file with the correct ordering.
def compute_elements_embeddings(elements_name):
    # Utils.init_logging('ComputeSentenceEmbeddings.log', logging.INFO)
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    pretrained_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                                    output_hidden_states=True)

    input_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.DENOMINATED + '_' + elements_name + ".h5")
    output_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.VECTORIZED + '_'  + Utils.DISTILBERT + '_'
                                   + elements_name + ".npy")
    vocabTable_db_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.INDICES_TABLE + ".sql")

    input_db = pd.HDFStore(input_filepath, mode='r')
    vocabTable_db = sqlite3.connect(vocabTable_db_filepath)
    vocabTable_db_c = vocabTable_db.cursor()

    matrix_of_sentence_embeddings = []
    vocabTable_db_c.execute("SELECT * FROM vocabulary_table")

    for row in vocabTable_db_c: # consuming the cursor iterator. Tuple returned: ('wide', 'adv.1', 2, 4, 5, 16, 18)
        sense_df = input_db.select(key=elements_name, where="word == " + str(row[0]) + " & sense =='" + row[1] + "'")
        element_text_series = sense_df[elements_name]
        for element_text in element_text_series:
            distilbert_sentence = compute_sentence_dBert_vector(pretrained_model, tokenizer, element_text).squeeze()
            sentence_d300 = compute_300d_sentence_vector(distilbert_sentence)
            matrix_of_sentence_embeddings.append(sentence_d300.squeeze().tolist())

    embds_nparray = np.array(matrix_of_sentence_embeddings)
    np.save(output_filepath, embds_nparray)
