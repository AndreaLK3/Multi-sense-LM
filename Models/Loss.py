import logging
import numpy as np
import torch
from torch.nn import functional as tfunc
from Models import ExplorePredictions as EP
from Utils import DEVICE, get_timestamp_month_to_sec
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import Filesystem as F

def write_doc_logging(model, model_forParameters):
    hyperparams_str = 'Model_' + model_forParameters.__class__.__name__ + '_date_' + get_timestamp_month_to_sec()
    logging.info("Hyperparameters: " + hyperparams_str)
    logging.info("Model:")
    logging.info(str(model))
    logging.info("Parameters:")
    parameters_list = [(name, param.shape, param.dtype, param.requires_grad) for (name, param) in model.named_parameters()]
    logging.info('\n'.join([str(p) for p in parameters_list]))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info("Number of trainable parameters=" + str(params))
    return hyperparams_str


def update_predictions_history_dict(predictions_history_dict, predictions_globals, predictions_senses,
                                    batch_labels_globals, batch_labels_all_senses, batch_labels_polysenses_dict):

    (probvalues_gs, predicted_gs) = predictions_globals.sort(dim=1, descending=True)

    predictions_history_dict['correct_g'] = \
        predictions_history_dict['correct_g'] + torch.sum(predicted_gs[:, 0] == batch_labels_globals).item()
    predictions_history_dict['tot_g'] = predictions_history_dict['tot_g'] + batch_labels_globals.shape[0]

    if len(predictions_senses.shape) > 1:
        (probvalues_s, predicted_s) = predictions_senses.sort(dim=1, descending=True)
        predictions_history_dict['correct_all_s'] = \
            predictions_history_dict['correct_all_s'] + torch.sum(predicted_s[:, 0] == batch_labels_all_senses).item()
        predictions_history_dict['tot_all_s'] = predictions_history_dict['tot_all_s'] + \
                                                (batch_labels_all_senses[batch_labels_all_senses != -1].shape[0])

        for threshold_key in batch_labels_polysenses_dict.keys():
            num_correct_polysenses = torch.sum(predicted_s[:, 0] == batch_labels_polysenses_dict[threshold_key]).item()
            predictions_history_dict['correct_poly_s'][threshold_key] = predictions_history_dict['correct_poly_s'][
                                                                       threshold_key] + num_correct_polysenses
            num_polysenses = batch_labels_polysenses_dict[threshold_key]\
                                                         [batch_labels_polysenses_dict[threshold_key] != -1].shape[0]
            predictions_history_dict['tot_poly_s'][threshold_key] = predictions_history_dict['tot_poly_s'][
                                                                       threshold_key] + num_polysenses

    logging.debug("updated_predictions_history_dict = " + str(predictions_history_dict))
    return

def organize_polysense_labels(batch_labels_globals, batch_labels_senses, polysense_globals_dict):
    # separately, for each threshold of polysemous senses:
    polysense_thresholds = polysense_globals_dict.keys()
    batch_labels_polysenses_dict = {}.fromkeys(polysense_thresholds)

    # init empty set
    for threshold_key in polysense_thresholds:
        batch_labels_polysenses_dict[threshold_key]=[]
    # categorize sense labels depending on the number of senses of their globalword
    for i in range(len(batch_labels_senses)):
        for threshold_key in polysense_thresholds:
            if batch_labels_globals[i].item() in polysense_globals_dict[threshold_key]:
                sense_label = batch_labels_senses[i]
                batch_labels_polysenses_dict[threshold_key].append(sense_label)
            else:
                batch_labels_polysenses_dict[threshold_key].append(-1)

    for threshold_key in polysense_thresholds:
        labels_ls = batch_labels_polysenses_dict[threshold_key]
        batch_labels_polysenses_dict[threshold_key] = torch.tensor(labels_ls).to(DEVICE)

    return batch_labels_polysenses_dict

def compute_model_loss(model, batch_input, batch_labels, correct_preds_dict, polysense_globals_dict,
                       vocab_sources_ls=[F.WT2, F.SEMCOR], sp_method=CE.Method.FASTTEXT, verbose=False):

    predictions_globals, predictions_senses = model(batch_input, batch_labels)

    batch_labels_t = (batch_labels).clone().t().to(DEVICE)
    batch_labels_globals = batch_labels_t[0]
    batch_labels_all_senses = batch_labels_t[1]

    # compute the loss for the batch
    loss_global = tfunc.nll_loss(predictions_globals, batch_labels_globals)

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__== "DataParallel" else model
    if model_forParameters.predict_senses:
        loss_all_senses = tfunc.nll_loss(predictions_senses, batch_labels_all_senses, ignore_index=-1)
    else:
        loss_all_senses = torch.tensor(0)
    # Adding accuracy to measure the senses' task, given that we can not rely on the senses' PPL for SelectK & co.
    batch_labels_polysenses_dict=organize_polysense_labels(batch_labels_globals, batch_labels_all_senses, polysense_globals_dict)
    update_predictions_history_dict(correct_preds_dict, predictions_globals, predictions_senses,
                                    batch_labels_globals, batch_labels_all_senses, batch_labels_polysenses_dict)

    # debug: check the solutions and predictions. Is there anything the model is unable to predict?
    if verbose:
        logging.info("*******\ncompute_model_loss > verbose logging of batch")
        EP.log_batch(batch_labels, predictions_globals, predictions_senses, 5, vocab_sources_ls, sp_method)

    losses_tpl = loss_global, loss_all_senses
    num_sense_instances = len(batch_labels_all_senses[batch_labels_all_senses != -1])

    return (losses_tpl, num_sense_instances)