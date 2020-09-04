import logging

import numpy as np
import torch
from torch.nn import functional as tfunc
from math import isnan, isinf
from NN import ExplorePredictions as EP
from Utils import DEVICE


def write_doc_logging(train_dataloader, model, model_forParameters, learning_rate, num_epochs):
    hyperparams_str = '_batchPerSeqlen' + str(train_dataloader.batch_size) \
                      + '_area' + str(model_forParameters.grapharea_size)\
                      + '_lr' + str(learning_rate) \
                      + '_epochs' + str(num_epochs)
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


def update_predictions_history_dict(correct_preds_dict, predictions_globals, predictions_senses, batch_labels_tpl):

    k = 10
    batch_labels_globals = batch_labels_tpl[0]
    batch_labels_all_senses = batch_labels_tpl[1]
    batch_labels_multi_senses = batch_labels_tpl[2]

    (values_g, indices_g) = predictions_globals.sort(dim=1, descending=True)

    correct_preds_dict['correct_g'] = \
        correct_preds_dict['correct_g'] + torch.sum(indices_g[:, 0] == batch_labels_globals).item()
    correct_preds_dict['tot_g'] = correct_preds_dict['tot_g'] + batch_labels_globals.shape[0]

    top_k_predictions_g = indices_g[:, 0:k]
    batch_counter_top_k_g = 0
    for i in range(len(batch_labels_globals)):
        label_g = batch_labels_globals[i]
        if label_g in top_k_predictions_g[i]:
            batch_counter_top_k_g = batch_counter_top_k_g+1
    correct_preds_dict['top_k_g'] = correct_preds_dict['top_k_g'] + batch_counter_top_k_g


    if len(predictions_senses.shape) > 1:
        (values_s, indices_s) = predictions_senses.sort(dim=1, descending=True)
        correct_preds_dict['correct_all_s'] = \
            correct_preds_dict['correct_all_s'] + torch.sum(indices_s[:, 0] == batch_labels_all_senses).item()
        correct_preds_dict['correct_multi_s'] = \
            correct_preds_dict['correct_multi_s'] + torch.sum(indices_s[:, 0] == batch_labels_multi_senses).item()
        correct_preds_dict['tot_all_s'] = correct_preds_dict['tot_all_s'] + \
                                      (batch_labels_all_senses[batch_labels_all_senses != -1].shape[0])
        correct_preds_dict['tot_multi_s'] = correct_preds_dict['tot_multi_s'] + \
                                          (batch_labels_all_senses[batch_labels_multi_senses != -1].shape[0])

        top_k_predictions_s = indices_s[:, 0:k]
        batch_counter_top_k_all_s = 0
        batch_counter_top_k_multi_s = 0
        for i in range(len(batch_labels_all_senses)):
            label_all_s = batch_labels_all_senses[i]
            label_multi_s = batch_labels_multi_senses[i]
            if label_all_s in top_k_predictions_s[i]:
                batch_counter_top_k_all_s = batch_counter_top_k_all_s + 1
            if label_multi_s in top_k_predictions_s[i]:
                batch_counter_top_k_multi_s = batch_counter_top_k_multi_s + 1
        correct_preds_dict['top_k_all_s'] = correct_preds_dict['top_k_all_s'] + batch_counter_top_k_all_s
        correct_preds_dict['top_k_multi_s'] = correct_preds_dict['top_k_multi_s'] + batch_counter_top_k_multi_s

    logging.debug("updated_predictions_history_dict = " +str(correct_preds_dict))
    return


def compute_model_loss(model, batch_input, batch_labels, correct_preds_dict, multisense_globals_set, verbose=False):

    predictions_globals, predictions_senses = model(batch_input)

    batch_labels_t = (batch_labels).clone().t().to(DEVICE)
    batch_labels_globals = batch_labels_t[0]
    batch_labels_all_senses = batch_labels_t[1]
    batch_labels_multi_senses_ls = list(map(
        lambda i : batch_labels_all_senses[i] if batch_labels_globals[i].item() in multisense_globals_set
                                              else -1, range(len(batch_labels_all_senses))))
    batch_labels_multi_senses = torch.tensor(batch_labels_multi_senses_ls).to(DEVICE)

    # compute the loss for the batch
    loss_global = tfunc.nll_loss(predictions_globals, batch_labels_globals)

    model_forParameters = model.module if torch.cuda.device_count() > 1 and model.__class__.__name__== "DataParallel" else model
    if model_forParameters.predict_senses:
        loss_all_senses = tfunc.nll_loss(predictions_senses, batch_labels_all_senses, ignore_index=-1)
        loss_multi_senses = tfunc.nll_loss(predictions_senses, batch_labels_multi_senses, ignore_index=-1)
    else:
        loss_all_senses = torch.tensor(0)
        loss_multi_senses = torch.tensor(0)
    # Added to measure the senses' task, given that we can not rely on the senses' PPL for SelectK
    batch_labels_tpl = (batch_labels_globals, batch_labels_all_senses, batch_labels_multi_senses)
    update_predictions_history_dict(correct_preds_dict, predictions_globals, predictions_senses, batch_labels_tpl)

    # debug: check the solutions and predictions. Is there anything the model is unable to predict?
    if verbose:
        logging.info("*******\ncompute_model_loss > verbose logging of batch")
        EP.log_batch(batch_labels, predictions_globals, predictions_senses, 2)

    losses_tpl = loss_global, loss_all_senses, loss_multi_senses
    senses_in_batch = len(batch_labels_all_senses[batch_labels_all_senses != -1])
    multisenses_in_batch = len(batch_labels_multi_senses[batch_labels_multi_senses != -1])
    num_sense_instances_tpl = senses_in_batch, multisenses_in_batch

    # Temp: debugging the GAT
    # logging.info("min & max (predictions_senses) = " + str(((torch.min(predictions_senses, dim=1)[0],
    #                                                             torch.max(predictions_senses, dim=1)[0]))))
    # logging.info("loss_global, loss_all_senses, loss_multi_senses= " + str((round(loss_global.item(),2),
    #                                                                         round(loss_all_senses.item(),2),
    #                                                                         round(loss_multi_senses.item(),2))))
    # logging.info("senses_in_batch, multisenses_in_batch= "+str((senses_in_batch, multisenses_in_batch)))
    # logging.info("Parameter | isfinite.all() | isnan.any()  | gradient.isfinite.all() | gradient.isnan.any()")
    # parameters_list = [(name, param.shape, param.dtype, torch.isfinite(param.data).all(), torch.isnan(param.data).any()) for (name, param) in
    #                   model.named_parameters()]
    # X_isnan = torch.isnan(model.X.data)
    # X_isinf = torch.isinf(model.X.data)
    # isnan_rows = [i for i in range(X_isnan.shape[0]) if True in X_isnan[i]]
    # logging.info('\n'.join([str(p) for p in parameters_list]))

    return (losses_tpl, num_sense_instances_tpl)