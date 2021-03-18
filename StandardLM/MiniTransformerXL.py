import transformers
import os
import Filesystem as F
import torch
import Utils
import StandardLM.TextCorpusReader as TCR
from math import ceil, inf, exp
import logging
import VocabularyAndEmbeddings.Vocabulary as V
import numpy as np
from time import time

# Auxiliary function: locate the file with the encoded corpus, and load it in chunks.
def get_numerical_corpus(corpus_name, split_name, vocabulary_sources_ls, chunk_size=1000):
    corpus_IDs_arr = TCR.load_corpus_IDs(corpus_name, split_name, vocabulary_sources_ls)
    corpus_total_size = corpus_IDs_arr.shape[0]
    num_chunks = ceil(corpus_total_size / chunk_size)
    corpus_chunks_ls = list(np.array_split(corpus_IDs_arr, num_chunks))
    return corpus_chunks_ls

# Auxiliary function: training loop during 1 epoch
def epoch_on_corpus(corpus_ids_chunks_ls, model, optimizer, training_or_test):
    if training_or_test:
        model.train()
    else:
        model.eval()

    total_loss = 0
    num_chunks = len(corpus_ids_chunks_ls)
    mems = None
    for i in range(num_chunks):
        input_ids = torch.tensor(corpus_ids_chunks_ls[i])
        enc_input = transformers.BatchEncoding()
        enc_input.data = {"input_ids":input_ids.unsqueeze(0)}
        enc_input = enc_input.to(Utils.DEVICE)
        lm_output_obj = model(input_ids=enc_input["input_ids"], mems=mems, labels=enc_input["input_ids"])
        all_losses = lm_output_obj.losses
        mems = lm_output_obj.mems
        loss = torch.mean(all_losses)
        total_loss = total_loss + loss.item()
        if training_or_test:
            loss.backward()
            optimizer.step()
        if i% (num_chunks//(min(100,num_chunks))) == 0:
            logging.info("Progress: " + str(round(i / (num_chunks),2) *100) + "% ...")

    epoch_loss = total_loss / num_chunks
    return epoch_loss


# Aims: - create a Transformer-XL model, smaller than the version applied on WT-103
#       - train the TXL model on the WikiText-2 dataset. Using: training split. Early stopping on validation, etc.
def txl_on_wt2(learning_rate=0.00001, max_num_epochs=500):
    Utils.init_logging("MiniTransformerXL-txl_on_wt2.log")
    vocab_filename = "vocabulary_" + F.WT2+ "_" + F.SEMCOR + ".txt"
    vocab_filepath = os.path.join(F.FOLDER_VOCABULARY, vocab_filename)
    # tokenizer = transformers.TransfoXLTokenizer(vocab_file=vocab_filepath)
    # if I read the pre-encoded corpus, I don't need the tokenizer

    vocab_df = V.get_vocabulary_df(corpora_names=[F.WT2, F.SEMCOR], lowercase=False)
    vocab_len = len(list(vocab_df["word"]))

    # we are going to extract and modify the configuration
    model_large = transformers.TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
    config = model_large.config

    # apply the necessary modifications
    config.n_head = 8       # from: 16
    config.n_layer = 12     # from: 18
    config.mem_len = 800    # from: 1600
    config.d_embed = 512    # from: 1024
    config.d_head = 64      # unchanged
    config.d_inner = 2048   # from: 4096
    config.d_model = 512    # from: 1024
    config.vocab_size = vocab_len # from: 267735
    config.adaptive = False # from: True ; whether to use Adaptive Softmax
    config.cutoffs = []
    config.div_val = 1

    model = transformers.TransfoXLLMHeadModel(config)
    model.to(Utils.DEVICE)
    optimizer = transformers.AdamW(model.parameters(), lr=learning_rate)

    wt2_train_chunks_ls = get_numerical_corpus(corpus_name=F.WT2, split_name=Utils.TRAINING,
                                                vocabulary_sources_ls=[F.WT2, F.SEMCOR], chunk_size=10) # chunk_size=1000
    wt2_valid_chunks_ls = get_numerical_corpus(corpus_name=F.WT2, split_name=Utils.VALIDATION,
                                               vocabulary_sources_ls=[F.WT2, F.SEMCOR], chunk_size=10) # chunk_size=1000
    # for testing purposes, works as using mini-corpora
    #wt2_train_chunks_ls = wt2_train_chunks_ls[0:10]
    #wt2_valid_chunks_ls = wt2_valid_chunks_ls[0:10]

    epoch = 1
    best_valid_loss = inf
    while epoch <= max_num_epochs:
        logging.info("Training: epoch n." + str(epoch) + "...")
        train_loss = epoch_on_corpus(wt2_train_chunks_ls, model, optimizer, training_or_test=True)
        train_ppl = exp(train_loss)
        logging.info("Epoch n." + str(epoch) + " completed. Training PPL=" + str(round(train_ppl, 2)))
        valid_loss = epoch_on_corpus(wt2_valid_chunks_ls, model, optimizer, training_or_test=False)
        logging.info("After epoch n." + str(epoch) + ", validation PPL=" + str(round(exp(valid_loss),2)))
        if (exp(valid_loss) > exp(best_valid_loss) + 0.1) and (epoch > 5):
            logging.info("Latest validation PPL of " + str(round(exp(valid_loss), 2))
                          + " > previous best validation PPL of " + str(round(exp(best_valid_loss), 2)))
            logging.info("Early stopping")
            break
        if exp(valid_loss) < (best_valid_loss):
            best_valid_loss = valid_loss
        epoch = epoch + 1

    torch.save(model, "MiniTXL_onWT2.pt")

