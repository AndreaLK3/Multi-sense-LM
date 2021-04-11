import transformers
import torch
from Models.TrainingSetup import setup_corpus, get_objects
from StandardLM.MiniTransformerXL import get_mini_txl_modelobj
import Utils
import os
import Filesystem as F

def miniexperiment_with_trainer(vocab_sources_ls=[F.WT2, F.SEMCOR], sp_method=Utils.SpMethod.FASTTEXT):
    Utils.init_logging("MiniExp_TXL_withTrainer.log")
    gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    objects = get_objects(vocab_sources_ls, sp_method, grapharea_size=32)

    model = get_mini_txl_modelobj(vocab_sources_ls)

    train_dataset, _ =\
        setup_corpus(objects, corpus_location=os.path.join(F.FOLDER_MINICORPORA, F.FOLDER_SENSELABELED, F.FOLDER_TRAIN),
                     slc_or_text=True, gr_in_voc_folders=gr_in_voc_folders,
                     batch_size=4, seq_len=5, model_forDataLoading=model)
    valid_dataset, _ =\
        setup_corpus(objects, corpus_location=os.path.join(F.FOLDER_MINICORPORA, F.FOLDER_SENSELABELED, F.FOLDER_VALIDATION),
                     slc_or_text=True, gr_in_voc_folders=gr_in_voc_folders,
                     batch_size=4, seq_len=5, model_forDataLoading=model)

    trainer = transformers.Trainer(model=model, train_dataset=train_dataset, eval_dataset=valid_dataset)

    trainer.train()
    trainer.evaluate(eval_dataset=valid_dataset)