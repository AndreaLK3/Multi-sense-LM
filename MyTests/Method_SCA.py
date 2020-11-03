import torch
import numpy as np
import pandas as pd
import Utils
import logging
from torch.nn.parameter import Parameter
from NN.Models.SenseContextAverage import update_context_average

def ctx_running_average(seq_len=3, bsz=2, dim=4, num_C=3):
    CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())
    torch.manual_seed(1)  # for reproducibility
    Utils.init_logging("Tests-SCA-ctx_running_average.log")
    logging.info("Hyperparameters: seq_len=" + str(seq_len)
                 + " ; batch size=" + str(bsz)
                 + " ; embedding dimension=" + str(dim)
                 + " ; num_C=" + str(num_C))

    prev_word_embeddings = torch.zeros(seq_len, bsz, dim)
    word_embeddings = torch.randint(low=-3, high=3, size=(seq_len, bsz, dim))
    logging.info("word_embeddings=" + str(word_embeddings))

    location_context = Parameter(torch.zeros(seq_len, bsz, dim))

    update_context_average(location_context, word_embeddings, prev_word_embeddings, num_C, CURRENT_DEVICE)
    logging.info("location_context=" + str(location_context))

    word_embeddings_2 = torch.randint(low=-3, high=3, size=(seq_len, bsz, dim))
    logging.info("word_embeddings_2=" + str(word_embeddings_2))

    update_context_average(location_context, word_embeddings_2, word_embeddings, num_C, CURRENT_DEVICE)
    logging.info("location_context=" + str(location_context))

def cosine_sim_ranking(seq_len=3, bsz=2, dim=4, senses_from_K=5):
    torch.manual_seed(1)
    Utils.init_logging("Tests-SCA-cosine_sim_ranking.log")

    cosine_sim = torch.nn.CosineSimilarity(dim=3)
    all_sense_neighbours = torch.randint(low=-21000, high=21000, size=(seq_len, bsz, senses_from_K))
    logging.info("all_sense_neighbours=" + str(all_sense_neighbours))
    senses_context = torch.randint(low=-2, high=2, size=(seq_len, bsz, senses_from_K, dim)).to(torch.float32)
    logging.info("senses_context=" + str(senses_context))
    location_context = torch.randint(low=-2, high=2, size=(seq_len, bsz, dim)).to(torch.float32)
    logging.info("location_context=" + str(location_context))

    samples_cosinesim = cosine_sim(location_context.unsqueeze(2), senses_context)
    logging.info("samples_cosinesim=" + str(samples_cosinesim))
    samples_sortedindices = torch.sort(samples_cosinesim, descending=True).indices
    logging.info("samples_sortedindices=" + str(samples_sortedindices))
    samples_firstindex = samples_sortedindices[:, :, 0]
    logging.info("samples_firstindex=" + str(samples_firstindex))
    samples_firstsense = torch.gather(all_sense_neighbours, dim=2, index=samples_firstindex.unsqueeze(2))
    logging.info("samples_firstsense=" + str(samples_firstsense))
