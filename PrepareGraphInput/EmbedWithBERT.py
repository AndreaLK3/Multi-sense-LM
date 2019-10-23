import os
import sqlite3
import numpy as np
import pandas as pd
import pytorch_transformers as pt
import torch
import logging

import Filesystem
import Utils

START_SENTENCE_TOKEN = "[CLS]"
END_SEP_TOKEN = "[SEP]"
DIMENSIONS = 768

def compute_sentence_vector(model, tokenizer, sentence_text):

    toks = tokenizer.tokenize(START_SENTENCE_TOKEN + sentence_text + END_SEP_TOKEN)
    indices = tokenizer.convert_tokens_to_ids(toks)

    segment_ids = [1] * len(indices)# single-sentence inputs only require a series of 1s
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(indices).unsqueeze(0)
    segment_tensor = torch.tensor(segment_ids).unsqueeze(0)

    with torch.no_grad():
        last_layer = model(tokens_tensor, segment_tensor)[0]
        # last_hidden_state: torch.FloatTensor of shape (batch_size, sequence_length, hidden_size) (here [1, 17, 768])

    #  The correct pooling strategy (mean, max, concatenation, etc.) and layers used (last four, all, last layer, etc.)
    #  are dependent on the application

    # To get a single vector for our entire sentence we have multiple application-dependent strategies,
    # a simple approach is to average the (/second-to-last) hidden layer of each token, producing one 768-length vector

    sentence_embedding = torch.mean(last_layer, dim=1)[0] # batch size 1
    logging.debug(sentence_embedding.shape)
    return sentence_embedding


# In the previous step, we stored the start-and-end indices in a Sqlite3 database.
# It is necessary to write the embeddings in the .npy file with the correct ordering.
def compute_sentence_embeddings(elements_name):
    #Utils.init_logging('ComputeSentenceEmbeddings.log', logging.INFO)
    model = pt.BertModel.from_pretrained('bert-base-uncased')
    tokenizer = pt.BertTokenizer.from_pretrained('bert-base-uncased')

    input_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.DENOMINATED + '_' + elements_name + ".h5")
    output_filepath = os.path.join(Filesystem.FOLDER_INPUT, Utils.VECTORIZED + '_' + elements_name + ".npy")
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
            matrix_of_sentence_embeddings.append(compute_sentence_vector(model, tokenizer, element_text).squeeze().tolist())

    embds_nparray = np.array(matrix_of_sentence_embeddings)
    np.save(output_filepath, embds_nparray)
