import torch
import numpy as np
import pandas as pd
import Utils
import logging
from torch.nn.parameter import Parameter
from NN.Models.SenseContextAverage import update_context_average

def ctx_running_average(seq_len=3, bsz=2, dim=4, num_C=3):
    torch.manual_seed(1)  # for reproducibility
    Utils.init_logging("Tests-SCA-running_context_average.log")
    logging.info("Hyperparameters: seq_len=" + str(seq_len)
                 + " ; batch size=" + str(bsz)
                 + " ; embedding dimension=" + str(dim)
                 + " ; num_C=" + str(num_C))

    prev_word_embeddings = torch.zeros(seq_len, bsz, dim)
    word_embeddings = torch.randint(low=-3, high=3, size=(seq_len, bsz, dim))
    logging.info("word_embeddings=" + str(word_embeddings))

    location_context = Parameter(torch.zeros(seq_len, bsz, dim))

    update_context_average(location_context, word_embeddings, prev_word_embeddings, num_C)
    logging.info("location_context=" + str(location_context))

    word_embeddings_2 = torch.randint(low=-3, high=3, size=(seq_len, bsz, dim))
    logging.info("word_embeddings_2=" + str(word_embeddings_2))

    update_context_average(location_context, word_embeddings_2, word_embeddings, num_C)
    logging.info("location_context=" + str(location_context))