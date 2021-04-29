import argparse
import Models.TrainingSetup as TS
import Models.TrainingAndEvaluation as TE
from time import time
import Utils
from Filesystem import get_model_name
import copy

def test():
    Utils.init_logging("Test_run_model_training.log")

    args = argparse.Namespace()
    args.model_type = "selectk"
    args.standard_lm = "transformer"
    args.use_graph_input = True,
    args.learning_rate = 5e-5
    args.num_epochs = 30
    args.sp_method = "fasttext"
    args.K = 1
    args.context_method = 0
    args.C = 20
    args.random_seed = 1

    args_to_load_standardlm = copy.deepcopy(args)
    args_to_load_standardlm.model_type = "standardlm"
    standardLM_model_fname = get_model_name(model=None, args=args_to_load_standardlm)
    standardLM_model = TS.load_model_from_file(standardLM_model_fname)

    t0 = time()
    model, train_dataloader, valid_dataloader = TS.setup_training_on_SemCor(standardLM_model, model_type=args.model_type,
                                 K=args.K, context_method_id=args.context_method, C=args.C,
                                 dim_qkv=300, grapharea_size=32, batch_size=3, seq_len=5)

    TE.run_train(model, train_dataloader, valid_dataloader,
                               learning_rate=args.learning_rate, num_epochs=args.num_epochs, predict_senses=True)

    # We also need to evaluate the model in question on SemCor's test set and on Raganato's SensEval benchmark,
    # but if the model has been saved that can be done later

    t1 = time() ; Utils.time_measurement_with_msg(t0, t1, "Trained model")