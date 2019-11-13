import torch
import torch.nn as nn
import transformers
import os
import Filesystem as F
import Utils
import logging
import numpy as np
import time

PADDED_WINDOW_SIZE = 256
BERT_DIMENSIONS = 768
TARGET_DIMENSIONS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextEncoder():
    def __init__(self, tokenizer, txt_fpath):
        self.tokenizer = tokenizer
        self.txt_fpath = txt_fpath
        self.i_pointer = 0
        self.prev_input_buffer = []
        self.txt_file = open(txt_fpath, encoding="utf-8", mode="r")

    def __next__(self):

        if len(self.prev_input_buffer) > 0:
            logging.debug("Retrieving the remaining " + str(len(self.prev_input_buffer)) + " tokens from the buffer")
            self.tokens = self.prev_input_buffer
            self.prev_input_buffer = [] # reset buffer
        else:
            self.text_line = self.txt_file.readline()
            if self.text_line == '':
                raise StopIteration
            self.tokens = self.tokenizer.tokenize(self.text_line)
            logging.debug(
                "Reading in next line.  " + str(len(self.tokens)) + " tokens")

        self.actual_length = min(len(self.tokens)+2, PADDED_WINDOW_SIZE) # used for the attention mask

        if len(self.tokens) > PADDED_WINDOW_SIZE - 2:
            logging.debug("Line length = " + str(len(self.tokens)) +
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
        self.layer_1 = nn.Linear(BERT_DIMENSIONS, (BERT_DIMENSIONS + TARGET_DIMENSIONS)//2) # 768 > 534
        self.layer_2 = nn.Linear((BERT_DIMENSIONS + TARGET_DIMENSIONS)//2, TARGET_DIMENSIONS) # 534 > 300

    def forward(self, bert_sentence):
        self.mid = self.layer_1(bert_sentence)
        self.sentence_d300 = self.layer_2(self.mid)

        return self.sentence_d300


class SentenceEmbeddingDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(TARGET_DIMENSIONS, (BERT_DIMENSIONS + TARGET_DIMENSIONS)//2) # 300 > 534
        self.layer_2 = nn.Linear((BERT_DIMENSIONS + TARGET_DIMENSIONS)//2, BERT_DIMENSIONS) # 534 > 768

    def forward(self, encoded_sentence):
        self.mid = self.layer_1(encoded_sentence)
        self.sentence_d768_reconstructed = self.layer_2(self.mid)

        return self.sentence_d768_reconstructed


class SentenceEmbeddingAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sentence_d768):
        self.sentence_d300 = self.encoder(sentence_d768)
        self.sentence_d768_reconstructed = self.decoder(self.sentence_d300)

        return self.sentence_d768_reconstructed



def evaluate(autoencoder_model, tokenizer, pretrained_model, valid_txt_fpath):

    pretrained_model.eval()
    autoencoder_model.eval()
    corpus_reader = TextEncoder(tokenizer, valid_txt_fpath)
    sentences_cosine_distance_ls = []

    try:
        step = 0
        while True:
            with torch.no_grad():
                encoded_window, nonpadded_length = corpus_reader.__next__() # Must refactor to eliminate code duplication
                nextline_input_ids = torch.tensor(encoded_window).unsqueeze(0).to(DEVICE)  # Batch size 1
                att_mask = torch.tensor([1] * nonpadded_length + [0] * (PADDED_WINDOW_SIZE - nonpadded_length)).to(DEVICE)
                dbert_outputs = pretrained_model(nextline_input_ids, attention_mask=att_mask)
                last_hidden_states = dbert_outputs[0]  # The last hidden-state is the first element of the output tuple
                sentence_embedding = torch.mean(last_hidden_states, dim=1)[0].unsqueeze(0)  # batch size 1

                autoenc_output = autoencoder_model(sentence_embedding)
                sentences_cosine_distance = 1 - nn.functional.cosine_similarity(sentence_embedding, autoenc_output)
                sentences_cosine_distance_ls.append(sentences_cosine_distance)
                distances_ls_np = list(map (lambda x: x.cpu().numpy(), sentences_cosine_distance_ls))
                step = step + 1

    except StopIteration:
        avg_distance_loss = np.average(np.array(distances_ls_np))

        logging.info("Cosine distance's reconstruction loss on the validation dataset = " + str(avg_distance_loss))
        return avg_distance_loss



def exe():
    Utils.init_logging(os.path.join(F.FOLDER_WORD_EMBEDDINGS,"AutoEncoderDBERT.log"))
    out_encoder_fpath = os.path.join(F.FOLDER_WORD_EMBEDDINGS, F.ENCODER_MODEL)

    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    pretrained_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True).to(DEVICE)
    pretrained_model.eval()

    encoder_model = SentenceEmbeddingEncoder().to(DEVICE)
    decoder_model = SentenceEmbeddingDecoder().to(DEVICE)
    my_autoencoder_model = SentenceEmbeddingAutoEncoder(encoder_model, decoder_model)

    loss_fn = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(my_autoencoder_model.parameters())

    train_fpath = os.path.join(F.FOLDER_WT2, F.WT_TRAIN_FILE)
    validation_txt_fpath = os.path.join(F.FOLDER_WT2, F.WT_VALID_FILE)
    corpus_reader = TextEncoder(tokenizer, train_fpath)

    num_epochs = 5
    steps_check = 500
    old_valid_loss = np.inf
    validation_alarm_previous_checkpoint = False

    for epoch in range(num_epochs):
        try:
            step = 0
            while True:
                with torch.enable_grad():

                    encoded_window, nonpadded_length = corpus_reader.__next__()
                    if nonpadded_length <= 3:
                        logging.debug("Almost-empty line. Skipping")
                        continue # I should skip almost-empty lines, like separators "***"

                    nextline_input_ids = torch.tensor(encoded_window).unsqueeze(0).to(DEVICE)  # Batch size 1
                    att_mask = torch.tensor([1] * nonpadded_length + [0] * (PADDED_WINDOW_SIZE - nonpadded_length)).to(DEVICE)
                    logging.debug("Attention mask shape: " + str(att_mask.shape) +
                     "; 1s = " + str(nonpadded_length) + " ; 0s= " + str((PADDED_WINDOW_SIZE - nonpadded_length)))
                    dbert_outputs = pretrained_model(nextline_input_ids, attention_mask=att_mask)
                    last_hidden_states = dbert_outputs[0]  # The last hidden-state is the first element of the output tuple
                    logging.debug(last_hidden_states.shape)

                    sentence_embedding = torch.mean(last_hidden_states, dim=1)[0].unsqueeze(0).to(DEVICE) # batch size 1

                    logging.debug("Dimension of (d)BERT sentence embedding = " + str(sentence_embedding.shape))

                    autoenc_output = my_autoencoder_model(sentence_embedding)
                    logging.debug("Dimension of autoencoded reconstructed sentence embedding = " + str(autoenc_output.shape))

                    loss = loss_fn(input1=autoenc_output, input2=sentence_embedding, target=torch.tensor(1, dtype=torch.float).to(DEVICE))
                    logging.debug("Cosine distance reconstruction loss = " + str(loss)  + "\n***\n")
                    loss.backward()
                    optimizer.step()  # Does the update
                    optimizer.zero_grad()  # zero the gradient buffers
                    step = step +1
                    if step % steps_check == 0:
                        logging.info("- GPU usage: " + str(Utils.get_gpu_memory_map()) + "***\n")

                    if step % steps_check == 0:
                        logging.info("Training the AutoEncoder. Checkpoint at " + str(steps_check) + "*n steps.")
                        logging.info("Epoch: " + str(epoch) + " ; step:" + str(step))
                        logging.debug("GPU usage, on each device, in MBs: " + str(Utils.get_gpu_memory_map()) + "\n***\n")

                        #Must: - save the model ; - evaluate average cosine distance on a validation set
                        valid_reconstruction_loss = evaluate(my_autoencoder_model, tokenizer,pretrained_model, validation_txt_fpath)
                        if valid_reconstruction_loss > old_valid_loss and not(validation_alarm_previous_checkpoint):
                            validation_alarm_previous_checkpoint = True
                        if valid_reconstruction_loss < old_valid_loss:
                            validation_alarm_previous_checkpoint = False

                        if valid_reconstruction_loss > old_valid_loss and validation_alarm_previous_checkpoint:
                            logging.info("The cosine-distance reconstruction loss on the validation set is= "
                                         + str(valid_reconstruction_loss))
                            logging.info("The previous validation loss was= " + str(old_valid_loss))
                            logging.info("Early-stopping.")
                            raise StopIteration

                        old_valid_loss = valid_reconstruction_loss

                        my_autoencoder_model.train()  # make the model trainable again
                        # since we are going to use this for inference, saving the state_dict is sufficient
                        torch.save(encoder_model.state_dict(), out_encoder_fpath)
                        torch.cuda.empty_cache()


        except StopIteration:
            # corpus pass finished. Must: - save the model ; - evaluate average cosine distance on a validation set
            evaluate(my_autoencoder_model, tokenizer, validation_txt_fpath)
            my_autoencoder_model.train() # make the model trainable again
            # since we are going to use this for inference, saving the state_dict is sufficient
            torch.save(encoder_model.state_dict(), out_encoder_fpath)
            torch.cuda.empty_cache()

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
