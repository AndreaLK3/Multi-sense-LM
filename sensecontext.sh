python run_model_training.py --model_type=sensecontext --standard_lm=gru --K=5;
python run_model_evaluation.py --model_type=sensecontext --standard_lm=gru --K=5;
python run_model_training.py --model_type=sensecontext --standard_lm=gru --K=5 --use_graph_input=True;
python run_model_evaluation.py --model_type=sensecontext --standard_lm=gru --K=5 --use_graph_input=True;