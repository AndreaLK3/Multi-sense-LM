import torch
import torch.nn as nn
import transformers
import os
import Filesystem as F
import Utils
import logging


PADDED_WINDOW_SIZE = 128
BERT_DIMENSIONS = 768
TARGET_DIMENSIONS = 300

class TextEncoder():
    def __init__(self, tokenizer, txt_fpath):
        self.tokenizer = tokenizer
        self.txt_fpath = txt_fpath
        self.i_pointer = 0
        self.prev_input_buffer = []
        self.txt_file = open(txt_fpath, encoding="utf-8", mode="r")

    def __next__(self):

        if len(self.prev_input_buffer) > 0:
            logging.info("Retrieving the remaining " + str(len(self.prev_input_buffer)) + " tokens from the buffer")
            self.tokens = self.prev_input_buffer
            self.prev_input_buffer = [] # reset buffer
        else:
            self.text_line = self.txt_file.readline()
            if self.text_line == '':
                raise StopIteration
            self.tokens = self.tokenizer.tokenize(self.text_line)
            logging.info(
                "Reading in next line.  " + str(len(self.tokens)) + " tokens")

        self.actual_length = min(len(self.tokens)+2, PADDED_WINDOW_SIZE) # used for the attention mask

        if len(self.tokens) > PADDED_WINDOW_SIZE - 2:
            logging.info("Line length = " + str(len(self.tokenized_text)) +
                         ". Sending the part beyond " + str(PADDED_WINDOW_SIZE  -2) + " to the reading buffer.")
            self.prev_input_buffer = self.tokens[self.i_pointer + PADDED_WINDOW_SIZE - 2:]


        if len(self.tokens) < PADDED_WINDOW_SIZE - 2:
            self.tokens = self.tokens + ['[PAD]'] * (PADDED_WINDOW_SIZE - 2 - len(self.tokens))

        self.tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokens)


        self.encoded_window_01 = self.tokenized_text[self.i_pointer:self.i_pointer + PADDED_WINDOW_SIZE - 2]
        # adding the 2 tokens [CLS] and [SEP] here
        self.encoded_window_02 = self.tokenizer.build_inputs_with_special_tokens(self.encoded_window_01)

        return self.encoded_window_02, self.actual_length



class SentenceEmbeddingEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.Linear(BERT_DIMENSIONS, TARGET_DIMENSIONS)

    def forward(self, bert_sentence):
        self.sentence_d300 = self.encoder_layer(bert_sentence)

        return self.sentence_d300


class SentenceEmbeddingDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder_layer = nn.Linear(TARGET_DIMENSIONS, BERT_DIMENSIONS)

    def forward(self, encoded_sentence):
        self.sentence_d768_reconstructed = self.decoder_layer(encoded_sentence)

        return self.sentence_d768_reconstructed


class SentenceEmbeddingAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SentenceEmbeddingEncoder()
        self.decoder = SentenceEmbeddingDecoder()

    def forward(self, sentence_d768):
        self.sentence_d300 = self.encoder(sentence_d768)
        self.sentence_d768_reconstructed = self.decoder(self.sentence_d300)

        return self.sentence_d768_reconstructed



def exe():
    Utils.init_logging(os.path.join("PrepareGraphInput","TuneDBert.log"))

    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    pretrained_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
    my_autoencoder_model = SentenceEmbeddingAutoEncoder()

    loss_fn = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(my_autoencoder_model.parameters())

    train_fpath = os.path.join(F.FOLDER_WT2, F.WT_TRAIN_FILE)
    corpus_reader = TextEncoder(tokenizer, train_fpath)

    num_epochs = 2

    for epoch in range(num_epochs):
        while True:
            try:
                encoded_window, nonpadded_length = corpus_reader.__next__()
                nextline_input_ids = torch.tensor(encoded_window).unsqueeze(0)  # Batch size 1
                att_mask = torch.tensor([1] * nonpadded_length + [0] * (PADDED_WINDOW_SIZE - nonpadded_length))
                logging.info("Attention mask shape: " + str(att_mask.shape) +
                             "; 1s = " + str(nonpadded_length) + " ; 0s= " + str((PADDED_WINDOW_SIZE - nonpadded_length)))

                outputs = pretrained_model(nextline_input_ids, attention_mask=att_mask)
                last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
                logging.info(last_hidden_states.shape)

                sentence_embedding = torch.mean(last_hidden_states, dim=1)[0].unsqueeze(0)  # batch size 1
                logging.info("Dimension of (d)BERT sentence embedding = " + str(sentence_embedding.shape))

                autoenc_output = my_autoencoder_model(sentence_embedding)
                logging.info("Dimension of autoencoded reconstructed sentence embedding = " + str(autoenc_output.shape))

                loss = loss_fn(input1=autoenc_output, input2=sentence_embedding, target=torch.tensor(1))
                logging.info("Cosine distance reconstruction loss = " + str(loss)  + "\n***\n")
                loss.backward()
                optimizer.step()  # Does the update
                optimizer.zero_grad()  # zero the gradient buffers

            except StopIteration:

                # corpus pass finished. Must: - save the model ; - evaluate average cosine distance on a validation set



    # n: I should consider the sentence only if it is len > 3, [CLS] + more than 1 element + [SEP]



    return corpus_reader






##### Previously used...
def mask_last_token(inputs, tokenizer):
    """ Mask the last token preparing inputs and labels for standard language modeling. """
    labels = inputs.clone()
    rows = labels.shape[0]
    columns = labels.shape[1]
    mask_formasking = torch.tensor([[False if col != columns - 1 else True for col in range(columns)] for _row in range(rows)])
    labels[~mask_formasking] = -1  # We only compute loss on masked tokens
    print(labels)

    # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    inputs[mask_formasking] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels
