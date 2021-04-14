import argparse
import Models.TrainingSetup as TS
import Models.TrainingAndEvaluation as TE
from time import time
import Utils
from Filesystem import get_model_name
import copy

def parse_training_arguments():

    parser = argparse.ArgumentParser(description='Creating a model, training it on the sense-labeled corpus.')
    # Necessary parameters
    parser.add_argument('--model_type', type=str, choices=['rnn', 'selectk', 'mfs', 'sensecontext', 'selfatt'],
                        help='model to use for Multi-sense Language Modeling')
    parser.add_argument('--standard_lm', type=str, choices=['gru', 'transformer', 'gold_lm'],
                        help='Which pre-trained instrument to load for standard Language Modeling subtask: '
                             'GRU, Transformer-XL, or reading ahead the correct next word')

    # Optional parameters
    parser.add_argument('--use_graph_input', type=bool, default=False,
                        help='Whether to use the GNN input from the dictionary graph alongside the pre-trained word'
                             ' embeddings.')
    parser.add_argument('--learning_rate', type=float, default=0.00005,
                        help='learning rate for training the model (it is a parameter of the Adam optimizer)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='maximum number of epochs for model training. It generally stops earlier because it uses '
                             'early-stopping on the validation set')
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
    parser.add_argument('--random_seed', type=int, default=0,
                        help='We can specify a random seed != 0 for reproducibility')

    args = parser.parse_args()
    return args


args = parse_training_arguments()
Utils.init_logging("starting_run_model_training.log")

args_to_load_standardlm = copy.deepcopy(args)
args_to_load_standardlm.model_type = "standardlm"
standardLM_model_fname = get_model_name(model=None, args=args_to_load_standardlm)
standardLM_model = TS.load_model_from_file(standardLM_model_fname)

t0 = time()

model, train_dataloader, valid_dataloader = TS.setup_training_on_SemCor(standardLM_model, model_type=args.model_type,
                             K=args.K, context_method_id=args.context_method, C=args.C,
                             dim_qkv=300, grapharea_size=32, batch_size=1, seq_len=4)

final_model = TE.run_train(model, train_dataloader, valid_dataloader,
                           learning_rate=args.learning_rate, num_epochs=args.num_epochs, predict_senses=True)

# We also need to evaluate the model in question on SemCor's test set and on Raganato's SensEval benchmark,
# but if the model has been saved that can be done later

t1 = time() ; Utils.time_measurement_with_msg(t0, t1, "Trained model")