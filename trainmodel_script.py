import CreateGraphInput as CGI
import WordEmbeddings.ComputeEmbeddings as CE
import Graph.DefineGraph as DG
import GNN.Training as T
import Utils
import logging
import GNN.SenseLabeledCorpus as SLC
import torch
import os
import sqlite3
import Graph.Adjacencies as AD




CGI.exe_from_input_to_vectors(do_reset=True, compute_single_prototype=True, sp_method=CE.Method.FASTTEXT,vocabulary_from_senselabeled=True, min_count=2)

DG.get_graph_dataobject(new=True, method=CE.Method.FASTTEXT)

model, train_dataloader, valid_dataloader = T.training_setup(slc_or_text_corpus=True, include_senses = True,
                                                             method=CE.Method.FASTTEXT, grapharea_size=32,
                                                             batch_size=40, sequence_length=35)

T.training_loop(model, 0.0001, train_dataloader, valid_dataloader, 100)

