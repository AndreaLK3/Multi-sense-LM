## Multi-sense Language Modeling

This repository contains the code to train Multi-sense language models 
on the sense-labeled SemCor corpus, optionally relying on a dictionary graph built using WordNet glosses.

Multi-sense language modeling specifies not only the next predicted word but also the sense it 
assumes in the context, using the labels found in WordNet 3.0.

### Software dependencies
The code was created and tested on Python 3.6.9, with Pytorch 1.5.0+cu101 and CUDA 10.1 

Note: a GPU/CUDA is optional, and PyTorch 1.5 with CPU also works; in that case a 
different version of some dependencies must be used, like for PyTorch-geometric 

The list of the necessary modules follows, including the versions used in this repository.
Different versions may work, but it's not guaranteed
```
fasttext == 0.9.2         
gensim == 3.8.3
h5py == 2.10.0
langid == 1.1.6
lxml == 4.5.0
nltk == 3.5
numpy == 1.18.4
pandas == 1.0.3
scikit-image == 0.16.2
scikit-learn == 0.23.0
scipy == 1.4.1   
tables == 3.6.1
torch == 1.5.0+cu101
torch-cluster == 1.5.4
torch-geometric == 1.4.3
torch-scatter == 2.0.4
torch-sparse == 0.6.2
torch-spline-conv == 1.2.0
transformers == 2.9.0
```

### Training a multi-sense language model

#### 1) Loading pre-trained vectors
`./get_pretrained_vectors.sh` <br/>
Preliminary step: download the English pre-trained FastText vectors (7.2 GB when uncompressed) found at https://fasttext.cc/docs/en/crawl-vectors.html

#### 2) Gathering and pre-processing input
`python run_prepare_input.py` <br/>
Preliminary step: read the sense-labeled corpus, find and store the corresponding glosses from WordNet and create the dictionary graph.<br/> 
Command parameters:
- `--grapharea_size` (int, default 32), to change the size of the graph mini-batch

#### 3) Creating and training a model
`python run_model_training.py --model_type rnn/selectk/mfs/sensecontext/selfatt` <br/>
Create the specified multi-sense language model, and train it on the included SemCor corpus obtaining perplexity and accuracy scores.
Command parameters:
- mandatory: `--model_type`, type=str, choices=['rnn', 'selectk', 'mfs', 'sensecontext', 'selfatt']
- `--learning_rate`, type=float, default=0.00005
- `--num_epochs`, type=int, default=24 
- `--use_graph_input`, type=str, default='no', choices=['no', 'concat', 'replace'] 
- `--K`, type=int, default=1 
- `--context_method`, type=int, default=0 
- `--C`, type=int, default=20,
- `--dim_qkv`, type=int, default=300  
- `--random_seed`, type=int, default=0

Once it encounters early-stopping on the validation set, a model is saved in the NN folder.

#### Examples
A SelectK model with K=1 uses 2 GRUs and picks the sense among those of the most likely K=1 word
predicted by the standard language modeling sub-task. To create it and train it, we execute step 3) as follows: <br/>
`python run_model_training.py --model_type selectk --K 1` <br/>
If we wish to concatenate the input signal computed by the GNN on the dictionary graph:
`python run_model_training.py --model_type selectk --K 1 --use_graph_input concat`

SenseContext model, choosing among the senses of the most likely K=5 words by comparing their sense context to the
current context. The representation of the current context is obtained via a 3-layer GRU:
`python run_model_training.py --model_type sensecontext --K 5 --context_method 1`

### Source code structure
- **Top-level files** <br/>
    Scripts and high-level code, general utilities, reading a sense-labeled corpus.
- Folder: **VocabularyAndEmbeddings** <br/>
    Given a sense-labeled corpus, create the vocabulary. Ways to access pre-trained single-prototype embeddings.
- Folder: **GetKBInputData** <br/>
    Given a vocabulary, find the corresponding WordNet synsets, store the glosses (definitions, examples), synonyms and antonyms
- Folder: **Graph** <br/>
    Given a vocabulary and WordNet data, create a dictionary graph with word, sense and gloss nodes.
- Folder: **NN** <br/>
    Given a sense-labeled corpus and a dictionary graph, create and train the specified multi-sense language model
- Folder: **TextCorpuses** <br/>
    Contains the SemCor sense-labeled corpus, and its splits.
- Folder: **MyTests** <br/>
    Development tests for features and models.
