import os

FOLDER_INPUT = 'InputData'
FOLDER_PCA = 'PCA' # subfolder of InputData

FOLDER_VOCABULARY = 'VocabularyAndEmbeddings'

FOLDER_TEXT_CORPORA = 'TextCorpora'
FOLDER_SENSELABELED = "SenseLabeled"
FOLDER_STANDARDTEXT = "StandardText"
FOLDER_MYTESTS = "MyTests"
FOLDER_MINICORPORA = 'MiniCorpora'

FOLDER_WT103 = 'wikitext-103'
FOLDER_WT2 = 'wikitext-2'
FOLDER_SEMCOR = 'semcor'

WT_TRAIN_FILE = 'wiki.train.tokens'
WT_VALID_FILE = 'wiki.valid.tokens'
WT_TEST_FILE = 'wiki.test.tokens'
CORPUS_NUMERICAL_EXTENSION = ".numerical."

FOLDER_TRAIN = 'Training'
FOLDER_VALIDATION = 'Validation'
FOLDER_TEST= 'Test'

WT2 = "WT2"
WT103 = "WT103"
SEMCOR = "SemCor"
CORPORA_LOCATIONS = {WT2: os.path.join(FOLDER_TEXT_CORPORA, FOLDER_STANDARDTEXT, FOLDER_WT2),
                     WT103: os.path.join(FOLDER_TEXT_CORPORA, FOLDER_STANDARDTEXT, FOLDER_WT103),
                     SEMCOR: os.path.join(FOLDER_TEXT_CORPORA, FOLDER_SENSELABELED, FOLDER_SEMCOR)
                     }

FOLDER_GRAPH = "Graph"
FOLDER_MODELS = 'Models'
FOLDER_SAVEDMODELS = "SavedModels"

BN_WORD_INTROS = 'BabelNet_word_intros'
BN_SYNSET_DATA = 'BabelNet_synset_data'
BN_SYNSET_EDGES = 'BabelNet_synset_edges'


VOCAB_CURRENT_INDEX_FILE = 'vocabulary_currentIndex.txt' # used for BabelNet requests over several days

FASTTEXT_PRETRAINED_EMBEDDINGS_FILE = 'cc.en.300.bin'
SPVs_FILENAME = "SinglePrototypeVectors.npy"

SEMCOR_DB = 'semcor_all.db'
MASC_DB = 'masc_written.db'

KBGRAPH_FILE = 'kbGraph.dataobject'
LOSSES_FILEEND = 'losses.npy'
PERPLEXITY_FILEEND = 'perplexity.npy'

GRAPHAREA_FILE = 'graphArea_matrix.npz'

MFS_H5_FPATH = os.path.join(FOLDER_TEXT_CORPORA, FOLDER_SENSELABELED, FOLDER_SEMCOR, FOLDER_TRAIN, "MostFrequentSense.h5")

MATRIX_SENSE_CONTEXTS_FILEEND = '_SenseContext.npy'

def get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method):
    inputdata_folder = set_directory(os.path.join(FOLDER_INPUT, "_".join(vocab_sources_ls), sp_method.value))
    graph_folder = set_directory(os.path.join(FOLDER_GRAPH, "_".join(vocab_sources_ls), sp_method.value))
    vocabulary_folder = set_directory(os.path.join(FOLDER_VOCABULARY, "_".join(vocab_sources_ls)))
    return graph_folder, inputdata_folder, vocabulary_folder

### Create the folder at a specified filepath, if it does not exist
def set_directory(dir_path):
    if os.path.exists(dir_path):
        return dir_path
    else:
        os.makedirs(dir_path)
    return dir_path