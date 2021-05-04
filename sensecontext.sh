echo proceeding to fine-tune models with Transformer standard language model;
python run_model_training.py --model_type=sensecontext --standard_lm=transformer --context_method_id=1 --pretrained_senses=True;
python run_model_training.py --model_type=sensecontext --standard_lm=transformer --context_method_id=1 --K=5 --pretrained_senses=True;
python run_model_training.py --model_type=sensecontext --standard_lm=transformer --context_method_id=1 --use_graph_input=True --pretrained_senses=True;