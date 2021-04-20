import torch
import Models.StandardLM.StandardLM as StandardLM
import Graph.PolysemousWords
import Utils
import Filesystem as F
import logging
import Graph.DefineGraph as DG
import os
import pandas as pd
import Graph.Adjacencies as AD
import numpy as np
from Utils import DEVICE
import Models.DataLoading.DataLoading as DL
import Models.Variants.RNNs as RNNs
from enum import Enum
import Models.Variants.SelectK as SelectK
import Models.Variants.SenseContext as SC
import Models.Variants.SelfAttention as SA
import Models.Variants.MFS as MFS
from Models.Variants.Common import ContextMethod

# ########## Auxiliary functions ##########

# ---------- a) Which variant of the model to use ----------
class ModelType(Enum):
    RNN = "RNN"
    SELECTK = "SelectK"
    SC = "Sense Context"
    SELFATT = "Self Attention Scores"
    MFS = "Most Frequent Sense"

# ---------- b) The option to load a pre-trained version of one of our models ----------
def load_model_from_file(filename):
    saved_model_path = os.path.join(F.FOLDER_SAVEDMODELS, filename)
    model = torch.load(saved_model_path) if torch.cuda.is_available() \
        else torch.load(saved_model_path, map_location=torch.device('cpu'))
    logging.info("Loading the model found at: " + str(saved_model_path))

    return model


# ---------- Step 1: setting up the graph, grapharea_matrix (used for speed) and the vocabulary  ----------
def get_objects(vocab_sources_ls, sp_method=Utils.SpMethod.FASTTEXT, grapharea_size=32):

    graph_folder, inputdata_folder, vocabulary_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    graph_dataobj = DG.get_graph_dataobject(False, vocab_sources_ls, sp_method).to(DEVICE)

    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, grapharea_size, hops_in_area=1, graph_folder=graph_folder)

    embeddings_matrix = torch.tensor(np.load(os.path.join(inputdata_folder, F.SPVs_FILENAME))).to(torch.float32)

    globals_vocabulary_fpath = os.path.join(vocabulary_folder, "vocabulary.h5")
    vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')

    vocabulary_numSensesList = vocabulary_df['num_senses'].to_list().copy()
    if all([num_senses == -1 for num_senses in vocabulary_numSensesList]):
        vocabulary_df = Graph.PolysemousWords.compute_globals_numsenses(graph_dataobj, grapharea_matrix, grapharea_size,
                                                                        inputdata_folder, globals_vocabulary_fpath)

    return graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix, inputdata_folder


# ---------- Step 2, auxiliary function: create the model for the StandardLM sub-task, whether GRU or T-XL ----------
def create_standardLM_model(objects, model_type, include_graph_input, batch_size):
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix, inputdata_folder = objects
    standardLM_model = StandardLM.StandardLM(graph_dataobj, grapharea_size, embeddings_matrix,
                 model_type, include_graph_input, vocabulary_df, batch_size)
    return standardLM_model


# ---------- Step 3, auxiliary function: create the senses' model, of the type we specify ----------
def create_model(model_type, standardLM_model, objects, K, context_method, C, dim_qkv, batch_size):
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix, inputdata_folder = objects
    if model_type.lower() == 'rnn':
        model = RNNs.RNN(standardLM_model, graph_dataobj, grapharea_size, grapharea_matrix,
                 vocabulary_df, batch_size=batch_size, n_layers=3, n_hid_units=1024)
    elif model_type.lower() == 'selectk':
        model = SelectK.SelectK(standardLM_model, graph_dataobj, grapharea_size, grapharea_matrix,
                 vocabulary_df, batch_size=batch_size, n_layers=3, n_hid_units=1024, K=K)
    elif model_type.lower() == 'sensecontext':
        model = SC.SenseContext(standardLM_model, graph_dataobj, grapharea_size, grapharea_matrix,
                 vocabulary_df, batch_size=batch_size, n_layers=3, n_hid_units=1024,
                 context_method=context_method, C=C, inputdata_folder=inputdata_folder, K=K)
    elif model_type.lower() == 'selfatt':
        model = SA.SelfAtt(standardLM_model, graph_dataobj, grapharea_size, grapharea_matrix,
                 vocabulary_df, batch_size=batch_size, n_layers=3, n_hid_units=1024,
                 context_method=context_method, C=C, inputdata_folder=inputdata_folder, dim_qkv=dim_qkv, K=K)
    elif model_type.lower() == 'mfs':
        mfs_df = pd.read_hdf(F.MFS_H5_FPATH)
        model = MFS.MFS(StandardLM, graph_dataobj, grapharea_size, grapharea_matrix,
                 vocabulary_df, batch_size=batch_size, n_layers=3, n_hid_units=1024, K=1, mfs_df=mfs_df)
    else:
        raise Exception ("Model type specification incorrect")
    return model


# ---------- Step 3: Creating the DataLoaders for training, validation and test datasets ----------
# Auxiliary function: get dataset and dataloader on a corpus, specifying filepath and type (slc vs. text)
def setup_corpus(objects, corpus_location, slc_or_text, gr_in_voc_folders, batch_size, seq_len):
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix, inputdata_folder = objects
    graph_folder, inputdata_folder, vocabulary_folder = gr_in_voc_folders

    bptt_collator = DL.BPTTBatchCollator(grapharea_size, seq_len)
    dataset = DL.TextDataset(corpus_location, slc_or_text, inputdata_folder, vocabulary_df,
                             grapharea_matrix, grapharea_size, graph_dataobj)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size * seq_len,
                                                   num_workers=0, collate_fn=bptt_collator)

    return dataset, dataloader

################
# Entry functions: get model, dataset and dataloader. Either a StandardLM model on WT2, or one of the
# model variants on SemCor
def setup_pretraining_on_WT2(model_type, include_graph_input, batch_size, seq_len,
                             vocab_sources_ls, sp_method, grapharea_size):

    gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    objects = get_objects(vocab_sources_ls, sp_method, grapharea_size)
    standardLM_model = create_standardLM_model(objects, model_type, include_graph_input, batch_size)
    standardLM_model.to(DEVICE)

    # corpus_train_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT2], F.WT_TRAIN_FILE)
    # corpus_valid_fpath = os.path.join(F.CORPORA_LOCATIONS[F.WT2], F.WT_VALID_FILE)
    # For mini-experiments:
    corpus_train_fpath = os.path.join(F.FOLDER_MYTESTS, F.FOLDER_MINICORPORA, F.FOLDER_STANDARDTEXT, F.WT_TRAIN_FILE)
    corpus_valid_fpath = os.path.join(F.FOLDER_MYTESTS, F.FOLDER_MINICORPORA, F.FOLDER_STANDARDTEXT, F.WT_VALID_FILE)

    slc_or_text = False
    train_dataset, train_dataloader = setup_corpus(objects, corpus_train_fpath, slc_or_text, gr_in_voc_folders,
                                                   batch_size, seq_len)
    valid_dataset, valid_dataloader = setup_corpus(objects, corpus_valid_fpath, slc_or_text, gr_in_voc_folders,
                                                   batch_size, seq_len)
    return standardLM_model, train_dataloader, valid_dataloader


def setup_training_on_SemCor(standardLM_model, model_type=None, K=1, context_method_id=0, C=20,
                             dim_qkv=300, grapharea_size=32, batch_size=32, seq_len=35,
                             vocab_sources_ls=[F.WT2, F.SEMCOR], sp_method=Utils.SpMethod.FASTTEXT):

    gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    objects = get_objects(vocab_sources_ls, sp_method, grapharea_size)

    context_method = ContextMethod.GRU if context_method_id==1 else ContextMethod.AVERAGE # if 0, the default
    model = create_model(model_type, standardLM_model, objects, K, context_method, C, dim_qkv, batch_size)
    model.to(DEVICE)
    standardLM_model.to(DEVICE)

    corpus_train_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], F.FOLDER_TRAIN)
    corpus_valid_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], F.FOLDER_VALIDATION)
    # For mini-experiments:
    # corpus_train_fpath = os.path.join(F.FOLDER_MYTESTS, F.FOLDER_MINICORPORA, F.FOLDER_SENSELABELED, F.FOLDER_TRAIN)
    # corpus_valid_fpath = os.path.join(F.FOLDER_MYTESTS, F.FOLDER_MINICORPORA, F.FOLDER_SENSELABELED, F.FOLDER_VALIDATION)
    slc_or_text = True

    train_dataset, train_dataloader = setup_corpus(objects, corpus_train_fpath, slc_or_text, gr_in_voc_folders,
                                                   batch_size, seq_len)
    valid_dataset, valid_dataloader = setup_corpus(objects, corpus_valid_fpath, slc_or_text, gr_in_voc_folders,
                                                   batch_size, seq_len)

    return model, train_dataloader, valid_dataloader
