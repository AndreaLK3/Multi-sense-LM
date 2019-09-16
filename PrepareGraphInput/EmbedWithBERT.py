import pytorch_transformers as pt
import torch
import logging
import Utils

START_SENTENCE_TOKEN = "[CLS]"
END_SEP_TOKEN = "[SEP]"


def get_sentence_embedding(model, tokenizer, sentence_text):

    toks = tokenizer.tokenize(START_SENTENCE_TOKEN + sentence_text + END_SEP_TOKEN)
    indices = tokenizer.convert_tokens_to_ids(toks)

    segment_ids = [1] * len(indices)# single-sentence inputs only require a series of 1s

    logging.info(toks)

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
    logging.info(sentence_embedding.shape)

    return sentence_embedding


def main():
    Utils.init_logging("EmbedWithBERT.log")
    model = pt.BertModel.from_pretrained('bert-base-uncased')
    tokenizer = pt.BertTokenizer.from_pretrained('bert-base-uncased')
    example_sentence = "Dictionary definitions and other elements must be converted into sentence embeddings."
    sent_vec = get_sentence_embedding(model, tokenizer, example_sentence)
    return sent_vec
