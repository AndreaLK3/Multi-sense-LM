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
FOLDER_SENSEVAL = 'senseval'

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
SENSEVAL = "SensEval"
CORPORA_LOCATIONS = {WT2: os.path.join(FOLDER_TEXT_CORPORA, FOLDER_STANDARDTEXT, FOLDER_WT2),
                     WT103: os.path.join(FOLDER_TEXT_CORPORA, FOLDER_STANDARDTEXT, FOLDER_WT103),
                     SEMCOR: os.path.join(FOLDER_TEXT_CORPORA, FOLDER_SENSELABELED, FOLDER_SEMCOR),
                     SENSEVAL: os.path.join(FOLDER_TEXT_CORPORA, FOLDER_SENSELABELED, FOLDER_SENSEVAL)
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

# Get the name of a model for loading/saving, using either a pre-existing model (accessing its attributes) or CLI arguments
def get_model_name_from_arguments(args):
    model_fname = args.model_type
    if args.standard_lm == "transformer":
        model_fname = model_fname + "_TransformerLM"
    elif args.standard_lm == "gold_lm":
        model_fname = model_fname + "_GoldLM"
    if args.use_graph_input is True:
        model_fname = model_fname + "_withGraph"

    if args.model_type not in ["rnn", "mfs", "standardlm"]:
        model_fname = model_fname + "_K" + str(args.K)
    if args.model_type  in ["sensecontext", "selfatt"]:
        model_fname = model_fname + "_C" + str(args.C)
        if args.context_method_id <= 0:
            context_method_name = "AVERAGE"
        else:
            context_method_name = "GRU"
        model_fname = model_fname + "_ctx" + str(context_method_name)
    return model_fname


def get_model_name(model, args):
    if model is None:
        return get_model_name_from_arguments(args)+".pt"
    model_type = model.__class__.__name__.lower()  # e.g. selectk
    model_fname = model_type

    if model.StandardLM.use_gold_lm:  # problem: this is in the StandardLM sub-object
            model_fname = model_fname + "_GoldLM"
    if model.StandardLM.use_transformer_lm:
            model_fname = model_fname + "_Transformer"

    if model.StandardLM.include_globalnode_input > 0:
            model_fname = model_fname + "_withGraph"

    if not (model.predict_senses) and model_type != "standardlm":
        model_fname = model_fname + "_noSenses"

    if model_type not in ["rnn", "mfs", "standardlm"]:
        model_fname = model_fname + "_K" + str(model.K)

    if model_type in ["sensecontext", "selfatt"]:
            model_fname = model_fname + "_C" + str(model.C)
            model_fname = model_fname + "_ctx" + str(model.context_method.name)

    return model_fname+".pt"

### Create the folder at a specified filepath, if it does not exist
def set_directory(dir_path):
    if os.path.exists(dir_path):
        return dir_path
    else:
        os.makedirs(dir_path)
    return dir_path


def get_standardLM_filename(args):
    model_type = "StandardLM_" + args.model_type.lower()
    model_name = model_type
    if args.use_graph_input:
        model_name = model_name + "_withGraph"

    return model_name + ".pt"