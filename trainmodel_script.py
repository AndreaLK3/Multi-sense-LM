import CreateGraphInput as CGI
import WordEmbeddings.ComputeEmbeddings as CE
import Graph.DefineGraph as DG
import GNN.Training as T
import Utils
import logging
import GNN.DataLoading as DL
import torch
import os
import sqlite3
import Graph.Adjacencies as AD



CGI.exe_from_input_to_vectors(do_reset=True, compute_single_prototype=True, sp_method=CE.Method.FASTTEXT,
                              vocabulary_from_senselabeled=False, min_count=2)

DG.get_graph_dataobject(new=True, method=CE.Method.FASTTEXT)

model, train_dataloader, valid_dataloader = T.training_setup(slc_or_text_corpus=False, include_senses = False,
                                                             method=CE.Method.FASTTEXT, grapharea_size=64,
                                                             hidden_state_dim=300, batch_size=None, sequence_length=35)

T.training_loop(model, 0.0001, train_dataloader, valid_dataloader, 100)

