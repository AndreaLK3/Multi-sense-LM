import transformers
from transformers import Trainer
import torch
from Models.TrainingSetup import setup_corpus, get_objects
from StandardLM.MiniTransformerXL import get_mini_txl_modelobj
import Utils
import os
import Filesystem as F
import Models.DataLoading as DL

# class MyTrainer(Trainer):
#     def __init__(self, model, train_dataset, valid_dataset, train_dataloader, valid_dataloader):
#          super(Trainer, self).__init__(self)
#          self.train_dataloader = train_dataloader
#          self.valid_dataloader = valid_dataloader


def miniexperiment_with_trainer(vocab_sources_ls=[F.WT2, F.SEMCOR], sp_method=Utils.SpMethod.FASTTEXT, seq_len=512):
    Utils.init_logging("MiniExp_TXL_withTrainer.log")
    gr_in_voc_folders = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    objects = get_objects(vocab_sources_ls, sp_method, grapharea_size=32)

    model = get_mini_txl_modelobj(vocab_sources_ls)

    # Datasets
    train_dataset, train_dataloader =\
        setup_corpus(objects, corpus_location=os.path.join(F.FOLDER_MINICORPORA, F.FOLDER_SENSELABELED, F.FOLDER_TRAIN),
                     slc_or_text=True, gr_in_voc_folders=gr_in_voc_folders,
                     batch_size=4, seq_len=5, model_forDataLoading=model)
    valid_dataset, valid_dataloader =\
        setup_corpus(objects, corpus_location=os.path.join(F.FOLDER_MINICORPORA, F.FOLDER_SENSELABELED, F.FOLDER_VALIDATION),
                     slc_or_text=True, gr_in_voc_folders=gr_in_voc_folders,
                     batch_size=4, seq_len=5, model_forDataLoading=model)
    # Trainer
    bptt_collator = DL.BPTTBatchCollator(grapharea_size=32, sequence_length=seq_len)
    trainer = transformers.Trainer(model=model, train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=bptt_collator)
    trainer.args.dataloader_pin_memory = False;

    trainer.train()
    trainer.evaluate(eval_dataset=valid_dataset)