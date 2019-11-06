import torch
import transformers
import os
import Filesystem as F
import Utils
import logging


PADDED_WINDOW_SIZE = 512
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
            self.tokenized_text = self.prev_input_buffer
        else:
            self.text_line = self.txt_file.readline()
            self.tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.text_line))

        if len(self.tokenized_text) > PADDED_WINDOW_SIZE:
            logging.info("Line length = " + str(len(self.tokenized_text)) +
                         ". Sending the part beyond " + str(PADDED_WINDOW_SIZE) + " to the reading buffer.")
            self.prev_input_buffer = self.tokenized_text[self.i_pointer + PADDED_WINDOW_SIZE:]

        self.encoded_window_01 = self.tokenized_text[self.i_pointer:self.i_pointer + PADDED_WINDOW_SIZE]
        self.encoded_window_02 = self.tokenizer.build_inputs_with_special_tokens(self.encoded_window_01)
        return self.encoded_window_02


def exe():
    Utils.init_logging(os.path.join("PrepareGraphInput","TuneDBert.log"))

    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    pretrained_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')

    train_fpath = os.path.join(F.FOLDER_WT2, F.WT_TRAIN_FILE)
    corpus_reader = TextEncoder(tokenizer, train_fpath)

    for i in range(5):
        nextline_input_ids = torch.tensor(corpus_reader.__next__()).unsqueeze(0)  # Batch size 1
        outputs = pretrained_model(nextline_input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        logging.info(last_hidden_states.shape)


    # n: I should consider the sentence only if it is len > 3, [CLS] + more than 1 element + [SEP]
    torch.nn.Linear((PADDED_WINDOW_SIZE * BERT_DIMENSIONS), ) # # Batch size 1

    return corpus_reader


# Previously used...
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
