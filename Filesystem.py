import os

FOLDER_INPUT = 'InputData'
FOLDER_WORD_EMBEDDINGS = 'WordEmbeddings'
FOLDER_VOCABULARY = 'Vocabulary'
FOLDER_TEXT_CORPUSES = 'TextCorpuses'
FOLDER_UFSAC = 'ufsac-public-2.1'
FOLDER_GRAPH = "Graph"
FOLDER_GRAPHNN = 'GNN'

BN_WORD_INTROS = 'BabelNet_word_intros'
BN_SYNSET_DATA = 'BabelNet_synset_data'
BN_SYNSET_EDGES = 'BabelNet_synset_edges'

FOLDER_WT103 = 'WikiText-103'
FOLDER_WT2 = 'wikitext-2'

WT_TRAIN_FILE = 'wiki.train.tokens'
WT_VALID_FILE = 'wiki.valid.tokens'
WT_TEST_FILE = 'wiki.test.tokens'

PHRASES_MODEL_FILE = 'phrases_model.pickle'
PHRASED_TRAINING_CORPUS = 'phrased_training_corpus.txt'
TEMPORARY_PHRASED_CORPUS = 'temp_phrased_training_corpus.txt'

VOCABULARY_OF_GLOBALS_FILE = 'vocabulary_of_globals.h5'
VOCAB_PHRASED = 'vocabulary_phrased.h5'

VOCAB_CURRENT_INDEX_FILE = 'vocabulary_currentIndex.txt' # used for BabelNet requests over several days

FASTTEXT_PRETRAINED_EMBEDDINGS_FILE = 'cc.en.300.bin'
SPVs_FASTTEXT_FILE = 'SinglePrototypes_withFastText.npy'
SPVs_DISTILBERT_FILE = 'SinglePrototypes_withDistilBERT.npy'

FOLDER_TRAIN = 'Training'
FOLDER_VALIDATION = 'Validation'
FOLDER_TEST= 'Test'

SEMCOR_DB = 'semcor_all.db'
MASC_DB = 'masc_written.db'

KBGRAPH_FILE = 'kbGraph.dataobject'
LOSSES_FILEEND = 'losses.npy'
PERPLEXITY_FILEEND = 'perplexity.npy'

GRAPHAREA_FILE = 'graphArea_matrix.npy'