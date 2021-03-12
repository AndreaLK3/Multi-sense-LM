import os

FOLDER_INPUT = 'InputData'
FOLDER_PCA = 'PCA' # subfolder of InputData

FOLDER_VOCABULARY = 'VocabularyAndEmbeddings'

FOLDER_TEXT_CORPUSES = 'TextCorpuses'
FOLDER_SENSELABELED = "SenseLabeled"
FOLDER_STANDARDTEXT = "StandardText"
FOLDER_MYTESTS = "MyTests"
FOLDER_MINICORPUSES = 'MiniCorpuses'

FOLDER_WT103 = 'wikitext-103'
FOLDER_WT2 = 'wikitext-2'
FOLDER_SEMCOR = 'semcor'

WT_TRAIN_FILE = 'wiki.train.tokens'
WT_VALID_FILE = 'wiki.valid.tokens'
WT_TEST_FILE = 'wiki.test.tokens'
CORPUS_NUMERICAL_EXTENSION = ".numerical.pt"

FOLDER_TRAIN = 'Training'
FOLDER_VALIDATION = 'Validation'
FOLDER_TEST= 'Test'

WT2 = "WT2"
WT103 = "WT103"
SEMCOR = "SemCor"
CORPORA_LOCATIONS = {WT2: os.path.join(FOLDER_TEXT_CORPUSES, FOLDER_STANDARDTEXT, FOLDER_WT2),
                     WT103: os.path.join(FOLDER_TEXT_CORPUSES, FOLDER_STANDARDTEXT, FOLDER_WT103),
                     SEMCOR: os.path.join(FOLDER_TEXT_CORPUSES, FOLDER_SENSELABELED, FOLDER_SEMCOR)
                     }

FOLDER_GRAPH = "Graph"
FOLDER_NN = 'NN'

BN_WORD_INTROS = 'BabelNet_word_intros'
BN_SYNSET_DATA = 'BabelNet_synset_data'
BN_SYNSET_EDGES = 'BabelNet_synset_edges'


VOCAB_CURRENT_INDEX_FILE = 'vocabulary_currentIndex.txt' # used for BabelNet requests over several days

FASTTEXT_PRETRAINED_EMBEDDINGS_FILE = 'cc.en.300.bin'
SPVs_FASTTEXT_FILE = 'SinglePrototypes_withFastText.npy'
SPVs_DISTILBERT_FILE = 'SinglePrototypes_withDistilBERT.npy'

SEMCOR_DB = 'semcor_all.db'
MASC_DB = 'masc_written.db'

KBGRAPH_FILE = 'kbGraph.dataobject'
LOSSES_FILEEND = 'losses.npy'
PERPLEXITY_FILEEND = 'perplexity.npy'

GRAPHAREA_FILE = 'graphArea_matrix.npz'

SAVED_MODEL_NAME = 'pretrained_model.pt'

MATRIX_SENSE_CONTEXTS_FILEEND = '_SenseContext.npy'
MOST_FREQ_SENSE_FILE = 'MostFrequentSense.h5'