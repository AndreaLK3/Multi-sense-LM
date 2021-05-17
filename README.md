## Multi-sense Language Modeling

This repository contains the code to train Multi-sense language models, that predict not only the next word but also the sense it 
assumes in the context, using labels from WordNet 3.0.<br/>
The part of the models that executes standard language modelling must be pre-trained on WikiText-2. 
The sense prediction architecture is then trained on the SemCor sense-labeled corpus. 
Optionally, it is possible to use an input signal from a dictionary graph with WordNet glosses, updated via a Graph Attention Network.

#### Example
As specified in the next section, the preliminary steps are: 1) loading FastText vectors, and 2) gathering and encoding WordNet
glosses for the vocabulary. Once they are done, a model can be:
- pre-trained on WikiText-2: <br/>
  `python run_model_pretraining.py --model_type=transformer`
- trained on SemCor: <br/>
  `python run_model_training.py --model_type=selectk --standard_lm=transformer --K=1`
- evaluated on the SemCor test split and the SemEval-SensEval dataset: <br/>
  `python run_model_evaluation.py --model_type=selectk --standard_lm=transformer --K=1`

If we wish to concatenate the input signal computed by the GNN on the dictionary graph,
it is necessary to add `--use_graph_input=True` to the commands above


### Training a multi-sense language model

#### 1) Loading pre-trained vectors
`./get_pretrained_vectors.sh` <br/>
Preliminary step: download the English pre-trained FastText vectors (7.2 GB when uncompressed) found at https://fasttext.cc/docs/en/crawl-vectors.html

#### 2) Gathering and pre-processing input
`python run_prepare_input.py` <br/>
Preliminary step: read the sense-labeled corpus, find and store the corresponding glosses from WordNet and create the dictionary graph.<br/> 
Command parameters:
- `--grapharea_size` (int, default 32), to change the size of the graph mini-batch

#### 3) Pre-training the Standard LM architecture
`python run_model_pretraining.py --model_type=gru/transformer/gold_lm` <br/>
Choose an architecture to handle the standard language modelling sub-task. The tool is pre-trained on WikiText-2.<br/>
Command parameters:
* mandatory:
    - `--model_type`, type=string, choices=gru, transformer, gold_lm <br/>
      Define the Standard LM tool: a 3-layer GRU, an 8-layer Transformer-XL, or a Gold LM that reads
ahead the label to output the correct prediction
* optional:       
    - `--use_graph_input`, type=bool. <br/>
    If adding `use_graph_input=True`, the Standard LM will make use of the graph input signal, computed by the Graph Attention Network on the dictionary graph
    - `--learning_rate`, type=float, default=5e-5. <br/>
    The learning rate used by the GRU. The Transformer-XL uses (1/2 * --learning_rate).
    - `--random_seed`, type=int, default=0. <br/> The experiments in the paper used random_seed=1
    
The standard language modelling architecture is saved in the SavedModels folder. E.g. *SavedModels/standardlm_transformer_withGraph.pt*. 
It will be loaded as needed.


#### 4) Creating and training a model
`python run_model_training.py --model_type=rnn/transformer/mfs/selectk/sensecontext/selfatt --standard_lm=gru/transformer/gold_lm` <br/>
Create the specified multi-sense language model, and train it on the SemCor corpus obtaining perplexity and accuracy scores. <br/>
Command parameters:
* mandatory:
     - `--model_type`, type=str, choices=rnn, transformer, selectk, mfs, sensecontext, selfatt <br/>
    The architecture for the sense prediction task
     - `--standard_lm`, type=str, choices = gru, transformer, gold_lm <br/>
    Which pre-trained architecture to use for the standard language modelling task.
* optional:
     - `--K`, type=int, default=1
     - `--context_method_id`, type=int, default=0 <br/>
       Which method to use to create the context representation for the SenseContext and Self-Attention architectures. <br/>
       0 = average (default), 1 = GRU
     - `--learning_rate`, type=float, default=5e-5
     - `--C`, type=int, default=20 <br/>
       How many tokens to average to create the context representation if context_method_id==0
     - `--random_seed`, type=int, default=0. <br/> The experiments in the paper used random_seed=1
    
Once it encounters early-stopping due to the senses' accuracy on the SemCor validation set, a model is saved in the SavedModels folder. <br/>
E.g. *SavedModels/selectk_transformer_withGraph_K1.pt*

#### 5) Evaluating a model on the test sets
`python run_model_evaluation.py --model_type=rnn/transformer/mfs/selectk/sensecontext/selfatt --standard_lm=gru/transformer/gold_lm` <br/>
Load a saved multi-sense language model, and evaluate it on the SemCor test split and the SensEval-SemEval dataset, obtaining perplexity and accuracy scores. <br/>
Command parameters: identical to 4)



### Software dependencies
The code was created and tested on Python 3.6.13, with Pytorch 1.5.0+cu101 and CUDA 10.1 

The list of the necessary modules follows, including the versions used in this repository.
Different versions may work, but it's not guaranteed
```    
gensim == 4.0.1
h5py == 3.1.0
langid == 1.1.6
lxml == 4.6.3
nltk == 3.6.2
numpy == 1.19.5
pandas == 1.1.5
scikit-image == 0.17.2
scikit-learn == 0.24.2
scipy == 1.5.4   
tables == 3.6.1
torch == 1.5.0+cu101
torch-cluster == 1.5.4
torch-geometric == 1.4.3
torch-scatter == 2.0.4
torch-sparse == 0.6.2
torch-spline-conv == 1.2.0
transformers == 4.5.1
```

### Source code structure
- **Top-level files** <br/>
    Scripts and general utilities
- Folder: **GetKBData** <br/>
    Given a vocabulary, find the corresponding WordNet synsets and retrieve the glosses (definitions, examples), synonyms and antonyms
- Folder: **Graph** <br/>
    Create a dictionary graph with word, sense and gloss nodes.   
- Folder: **InputData** <br/>
  Storage location for the glosses and their encodings
- Folder: **InputPipeline** <br/>
  The high-level code to read a sense-labeled corpus, create a Vocabulary, get WordNet data and encode it.
- Folder: **Models** <br/>
  Create, pre-train, train and evaluate a Multi-sense language model
  - Subfolders: Auxiliary, DataLoading, StandardLM, Variants
- Folder: **SavedModels** <br/>
  Storage location for saved models (whether standard LM after pre-training, or full Multi-sense LM after training)
- Folder: **TextCorpora** <br/>
    Contains the SemCor and SensEval-SemEval sense-labeled corpora.
- Folder: **VocabularyAndEmbeddings** <br/>
    Given a sense-labeled corpus, create the vocabulary. Moreover, access the pre-trained single-prototype embeddings.