import argparse
import Models.TrainingSetup as TS
import Models.TrainingAndEvaluation as TE
import torch
from time import time
import Utils
import VocabularyAndEmbeddings.ComputeEmbeddings as CE
import Filesystem as F

def parse_arguments():
    parser = argparse.ArgumentParser(description='Creating a model, training it on the sense-labeled corpus.')
    # Necessary parameters
    parser.add_argument('--model_type', type=str, choices=['rnn', 'selectk', 'mfs', 'sensecontext', 'selfatt'],
                        help='model to use for Multi-sense Language Modeling')

    # Optional parameters
    parser.add_argument('--learning_rate', type=float, default=0.00005,
                        help='learning rate for training the model (it is a parameter of the Adam optimizer)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='maximum number of epochs for model training. It generally stops earlier because it uses '
                             'early-stopping on the validation set')
    parser.add_argument('--use_graph_input', type=str, default='no',
                        choices=['no', 'concat', 'replace'],
                        help='Whether to use the GNN input from the dictionary graph alongside the pre-trained word'
                             ' embeddings.')
    parser.add_argument('--sp_method', type=str, default='fasttext', choices=['fasttext', 'transformer'],
                        help="Which method is used to create the single-prototype embeddings: FastText or Transformer")

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
    parser.add_argument('--random_seed', type=int, default=0,
                        help='We can specify a random seed != 0 for reproducibility')

    args = parser.parse_args()
    return args

def convert_arguments_into_parameters(arguments):
    parameters = dict()
    if arguments.model_type == 'rnn':
        parameters["model_type"] = TS.ModelType.RNN
    elif arguments.model_type == 'selectk':
        parameters["model_type"] = TS.ModelType.SELECTK
    elif arguments.model_type == 'mfs':
        parameters["model_type"] = TS.ModelType.MFS
    elif arguments.model_type == 'sensecontext':
        parameters["model_type"] = TS.ModelType.SC
    elif arguments.model_type == 'selfatt':
        parameters["model_type"] = TS.ModelType.SELFATT

    if arguments.use_graph_input == 'no':
        parameters["include_globalnode_input"] = 0
    elif arguments.use_graph_input == 'concat':
        parameters["include_globalnode_input"] = 1
    elif arguments.use_graph_input == 'replace':
        parameters["include_globalnode_input"] = 2

    if arguments.context_method == 0:
        parameters["context_method"] = TS.ContextMethod.AVERAGE
    elif arguments.context_method == 1:
        parameters["context_method"] = TS.ContextMethod.GRU

    return parameters

args = parse_arguments()

parameters = convert_arguments_into_parameters(args)

t0 = time()

model, train_dataloader, valid_dataloader = TS.setup_training_on_corpus(F.WT2,
                                                                        premade_model=None, model_type=parameters["model_type"],
                                                                        include_globalnode_input=parameters["include_globalnode_input"], use_gold_lm=False, K=args.K,
                                                                        sp_method=Utils.SpMethod.FASTTEXT, context_method=parameters["context_method"], C=args.C,
                                                                        dim_qkv=args.dim_qkv, grapharea_size=32, batch_size=32, seq_len=35,
                                                                        vocab_sources_ls=(F.WT2, F.SEMCOR), random_seed=1)

pretrained_model = TE.run_train(model, train_dataloader, valid_dataloader,
                                learning_rate=0.0001, num_epochs=args.num_epochs, predict_senses=False,  # pre-training on WT2, lr=1e-4
                                vocab_sources_ls=(F.WT2, F.SEMCOR), sp_method=Utils.SpMethod.FASTTEXT)

_, train_dataloader, valid_dataloader = TS.setup_training_on_corpus(F.SEMCOR,
  premade_model=pretrained_model)

final_model = TE.run_train(model, train_dataloader, valid_dataloader,
                           learning_rate=args.learning_rate, num_epochs=args.num_epochs, predict_senses=True,  # on SemCor
                           vocab_sources_ls=(F.WT2, F.SEMCOR), sp_method=Utils.SpMethod.FASTTEXT)

# We also need to evaluate the model in question on SemCor's test set and on Raganato's SensEval benchmark,
# but if the model has been saved that can be done later

t1 = time() ; Utils.time_measurement_with_msg(t0, t1, "Trained model")