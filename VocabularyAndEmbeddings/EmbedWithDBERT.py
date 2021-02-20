import os
import torch
import logging
import Filesystem

START_SENTENCE_TOKEN = "[CLS]"
END_SEP_TOKEN = "[SEP]"


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

