python run_model_training.py --model_type=selectk --standard_lm=gold_lm --K=5 --use_graph_input=True;
python run_model_training.py --model_type=selectk --standard_lm=transformer  --pretrained_senses=True;
python run_model_training.py --model_type=selectk --standard_lm=transformer --K=5 --pretrained_senses=True;
python run_model_training.py --model_type=selectk --standard_lm=transformer --use_graph_input=True --pretrained_senses=True;
python run_model_training.py --model_type=selectk --standard_lm=transformer --K=5 --use_graph_input=True --pretrained_senses=True;