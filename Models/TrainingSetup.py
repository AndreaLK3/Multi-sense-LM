import torch
import Utils
import Filesystem as F
import logging
import Graph.DefineGraph as DG
import sqlite3
import os
import pandas as pd
import Graph.Adjacencies as AD
import numpy as np
from Utils import DEVICE
import Models.DataLoading as DL
import Models.Variants.RNNs as RNNs
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
from enum import Enum
import Models.Variants.SelectK as SelectK
import Models.Variants.SenseContext as SC
import Models.Variants.SelfAttention as SA
import Models.Variants.MFS as MFS
from Models.Variants.Common import ContextMethod # for ease of launch

# ########## Auxiliary functions ##########

# ---------- a) Which variant of the model to use ----------
class ModelType(Enum):
    RNN = "RNN"
    SELECTK = "SelectK"
    SC = "Sense Context"
    SELFATT = "Self Attention Scores"
    MFS = "Most Frequent Sense"

# ---------- b) The option to load a pre-trained version of one of our models ----------
def load_model_from_file(modeltype):
    saved_model_path = os.path.join(F.FOLDER_MODELS, F.FOLDER_SAVEDMODELS, modeltype.value + ".pt")
    model = torch.load(saved_model_path).module if torch.cuda.is_available() \
        else torch.load(saved_model_path, map_location=torch.device('cpu')).module # unwrapping DataParallel
    logging.info("Loading the model found at: " + str(saved_model_path))

    return model


# ---------- Step 1: setting up the graph, grapharea_matrix (used for speed) and the vocabulary  ----------
def get_objects(vocab_sources_ls, sp_method=CE.Method.FASTTEXT, grapharea_size=32):

    graph_folder, inputdata_folder, vocabulary_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    graph_dataobj = DG.get_graph_dataobject(False, vocab_sources_ls, sp_method).to(DEVICE)

    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, grapharea_size, hops_in_area=1, graph_folder=graph_folder)

    embeddings_matrix = torch.tensor(np.load(os.path.join(inputdata_folder, F.SPVs_FILENAME))).to(torch.float32)

    globals_vocabulary_fpath = os.path.join(vocabulary_folder, "vocabulary.h5")
    vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')

    vocabulary_numSensesList = vocabulary_df['num_senses'].to_list().copy()
    if all([num_senses == -1 for num_senses in vocabulary_numSensesList]):
        vocabulary_df = AD.compute_globals_numsenses(graph_dataobj, grapharea_matrix, grapharea_size,
                                                     inputdata_folder, globals_vocabulary_fpath)

    return graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix


# ---------- Step 2, auxiliary function: create the model, of the type we specify ----------
def create_model(model_type, objects, use_gold_lm, include_globalnode_input, K, context_method, C, dim_qkv):
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix = objects
    if model_type==ModelType.RNN:
        model = RNNs.RNN(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                           embeddings_matrix, include_globalnode_input, use_gold_lm,
                           batch_size=32, n_layers=3, n_hid_units=1024)
    elif model_type==ModelType.SELECTK:
        model = SelectK.SelectK(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                                    use_gold_lm, include_globalnode_input, batch_size=32, n_layers=3, n_hid_units=1024, K=1)
    elif model_type==ModelType.SC:
        model = SC.SenseContext(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
                                embeddings_matrix, use_gold_lm, include_globalnode_input, batch_size=32, n_layers=3,
                                n_hid_units=1024, K=K, num_C=C, context_method=context_method)
    elif model_type==ModelType.SELFATT:
        model = SA.ScoresLM(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
             use_gold_lm, include_globalnode_input, batch_size=32, n_layers=3, n_hid_units=1024, K=K,
                            num_C=C, context_method=context_method, dim_qkv=dim_qkv)
    elif model_type==ModelType.MFS:
        mfs_df = pd.read_hdf(F.MFS_H5_FPATH)
        model = MFS.MFS(graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix,
                            use_gold_lm, include_globalnode_input, batch_size=32, n_layers=3, n_hid_units=1024, K=1, mfs_df=mfs_df)
    else:
        raise Exception ("Model type specification incorrect")
    return model


# ---------- Step 2: locating the necessary folders, creating the model, moving it on GPU if needed
def setup_model(model_type, include_globalnode_input, use_gold_lm, K,
                load_saved_model, sp_method, context_method, C,
                dim_qkv, grapharea_size, batch_size, vocab_sources_ls, random_seed=1):

    # -------------------- 1: Setting up the graph, grapharea_matrix and vocabulary --------------------
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix = get_objects(vocab_sources_ls, sp_method, grapharea_size)
    objects = graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix

    # -------------------- 2: Loading / creating the model --------------------
    if random_seed != 0:
        torch.manual_seed(random_seed) # for reproducibility while conducting experiments
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    if load_saved_model: # allows to load a model pre-trained on another dataset. Was not used for the paper results.
        model = load_model_from_file(model_type)
    else:

        model = create_model(model_type, objects, use_gold_lm, include_globalnode_input, K, context_method, C, dim_qkv)

    # -------------------- 3: Moving objects on GPU --------------------
    logging.info("Graph-data object loaded, model initialized. Moving them to GPU device(s) if present.")

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        logging.info("Using " + str(n_gpu) + " GPUs")
        model = torch.nn.DataParallel(model, dim=0)
        model_forDataLoading = model.module
        batch_size = n_gpu if batch_size is None else (batch_size + batch_size % n_gpu)
        # if not specified, default batch_size = n. GPUs. use_gold_lm needs batch_size % n_gpu == 0 to work correctly
    else:
        model_forDataLoading = model
        batch_size = 1 if batch_size is None else batch_size
    model.to(DEVICE)

    return model, model_forDataLoading, batch_size


# ---------- Step 3: Creating the DataLoaders for training, validation and test datasets ----------
# Auxiliary function: get dataset and dataloader on a corpus, specifying filepath and type (slc vs. text)
def setup_corpus(objects, corpus_location, slc_or_text, gr_in_voc_folders, batch_size, seq_len, model_forDataLoading):
    graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix = objects
    graph_folder, inputdata_folder, vocabulary_folder = gr_in_voc_folders

    bptt_collator = DL.BPTTBatchCollator(grapharea_size, seq_len)
    dataset = DL.TextDataset(corpus_location, slc_or_text, inputdata_folder, vocabulary_df,
                             model_forDataLoading, grapharea_matrix, grapharea_size, graph_dataobj)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size * seq_len,
                                                   num_workers=0, collate_fn=bptt_collator)

    return dataset, dataloader

################

def setup_training_on_semcor(model_type, include_globalnode_input, use_gold_lm, K,
                load_saved_model=False, sp_method=CE.Method.FASTTEXT, context_method=ContextMethod.AVERAGE, C=0,
                dim_qkv=300, grapharea_size=32, batch_size=32, seq_len=35, vocab_sources_ls=(F.WT2, F.SEMCOR), random_seed=1):
    gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)

    objects = get_objects(vocab_sources_ls, sp_method, grapharea_size)
    # objects == graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df, embeddings_matrix

    model, model_forDataLoading, batch_size = setup_model(model_type, include_globalnode_input, use_gold_lm, K,
                                                                load_saved_model, sp_method, context_method, C,
                                            dim_qkv, grapharea_size, batch_size, vocab_sources_ls, random_seed)

    semcor_train_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], F.FOLDER_TRAIN)
    semcor_valid_fpath = os.path.join(F.CORPORA_LOCATIONS[F.SEMCOR], F.FOLDER_VALIDATION)

    train_dataset, train_dataloader = setup_corpus(objects, semcor_train_fpath, True, gr_in_voc_folders,
                                                   batch_size, seq_len, model_forDataLoading)
    valid_dataset, valid_dataloader = setup_corpus(objects, semcor_valid_fpath, True, gr_in_voc_folders,
                                                   batch_size, seq_len, model_forDataLoading)
    return model, train_dataloader, valid_dataloader


