import argparse
import NN.Training as T
import Graph.Adjacencies as AD
from time import time
import Utils
import logging
import os
import Filesystem as F
import VocabularyAndEmbeddings.ComputeEmbeddings as CE

def parse_arguments():
    parser = argparse.ArgumentParser(description='Creating a model, training it on the sense-labeled corpus.')
    # Necessary parameters
    parser.add_argument('--model_type', type=str, choices=['rnn', 'selectk', 'mfs', 'sensecontext', 'selfatt'],
                        help='model to use for Multi-sense Language Modeling')

    # Optional parameters
    parser.add_argument('--learning_rate', type=float, default=0.00005,
                        help='learning rate for training the model (it is a parameter of the Adam optimizer)')
    parser.add_argument('--num_epochs', type=int, default=24,
                        help='maximum number of epochs for model training. It generally stops earlier because it uses '
                             'early-stopping on the validation set')
    parser.add_argument('--use_graph_input', type=str, default='no',
                        choices=['no', 'concat', 'replace'],
                        help='Whether to use the GNN input from the dictionary graph alongside the pre-trained word'
                             ' embeddings.')

    # Optional parameters that are method-specific
    parser.add_argument('--K', type=int, default=1,
                        help='we choose the correct senses among those of the first top-K predicted words')
    parser.add_argument('--context_method', type=int, default=0,
                        help='Which context representation to use, in the methods: SenseContext, Self-Attention scores.'
                             ' 0=average of the last C tokens; 1=GRU with 3 layers')
    parser.add_argument('--C', type=int, default=20,
                        help='number of previous tokens to average to get the context representation (if used)')
    parser.add_argument('--dim_qkv', type=int, default=300,
                        help='dimensionality of queries, keys & vectors for the Self-Attention Scores method')

    args = parser.parse_args()
    return args

def convert_arguments_into_parameters(arguments):
    parameters = dict()
    if arguments.model_type == 'rnn':
        parameters["model_type"] = T.ModelType.RNN
    elif arguments.model_type == 'selectk':
        parameters["model_type"] = T.ModelType.SELECTK
    elif arguments.model_type == 'mfs':
        parameters["model_type"] = T.ModelType.MFS
    elif arguments.model_type == 'sensecontext':
        parameters["model_type"] = T.ModelType.SC
    elif arguments.model_type == 'selfatt':
        parameters["model_type"] = T.ModelType.SELFATT

    if arguments.use_graph_input == 'no':
        parameters["include_globalnode_input"] = 0
    elif arguments.use_graph_input == 'concat':
        parameters["include_globalnode_input"] = 1
    elif arguments.use_graph_input == 'replace':
        parameters["include_globalnode_input"] = 2

    if arguments.context_method == 0:
        parameters["context_method"] = T.ContextMethod.AVERAGE
    elif arguments.context_method == 1:
        parameters["context_method"] = T.ContextMethod.GRU

    return parameters

args = parse_arguments()

parameters = convert_arguments_into_parameters(args)

t0 = time()
model, datasets, dataloaders = T.setup_train(
        slc_or_text_corpus=True,
        model_type = parameters["model_type"],
        K=args.K,
        C=args.C,
        context_method=parameters["context_method"],
        dim_qkv=args.dim_qkv,
        include_globalnode_input=parameters["include_globalnode_input"],
        load_saved_model=False,
        batch_size=32, sequence_length=35,
        method=CE.Method.FASTTEXT,
        grapharea_size=32)
T.run_train(model, dataloaders, learning_rate=args.learning_rate, num_epochs=args.num_epochs)
t1 = time() ; Utils.time_measurement_with_msg(t0, t1, "Trained model")