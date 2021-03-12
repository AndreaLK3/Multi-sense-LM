import transformers
import os
import Filesystem as F
import torch
import Utils
import StandardLM.TextCorpusReader as TCR

# Auxiliary function: locate the file with the encoded corpus, and load it in chunks.
def load_numerical_corpus(corpus_name, split_name, vocabulary_sources_ls, chunk_size=1000):
    numerical_IDs_t = TCR.load_corpus_IDs(corpus_name, split, vocabulary_sources_ls)


# Aims: - create a Transformer-XL model, smaller than the version applied on WT-103
#       - train the TXL model on the WikiText-2 dataset. Using: training split. Early stopping on validation, etc.
def txl_on_wt2(learning_rate=0.0001, num_epochs=10):

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

    model = transformers.TransfoXLLMHeadModel(config)

    model.train()  # To train the model, we should first set it back in training mode
    optimizer = transformers.AdamW(model.parameters(), lr=learning_rate)

    epoch = 1
    while epoch <= num_epochs:

        print("***\nPrediction task: Training the ALBERT model, epoch n.:" + str(epoch))
        # epoch_loss = training_epoch(training_pairs, labels, model, tokenizer, args.batch_size, optimizer)
        epoch = epoch + 1
        print("Epoch n." + str(epoch) + " completed. Training loss=" + str(round(epoch_loss, 2)))

    torch.save(model, "MiniTXL_onWT2.pt")

