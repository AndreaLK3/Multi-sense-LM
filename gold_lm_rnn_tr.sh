python run_model_training.py --model_type=rnn --standard_lm=gold_lm;
python run_model_training.py --model_type=transformer --standard_lm=gold_lm;
python run_model_evaluation.py --model_type=rnn --standard_lm=gold_lm;
python run_model_evaluation.py --model_type=transformer --standard_lm=gold_lm;