import os

FOLDER_INPUT = 'InputData'
FOLDER_WORD_EMBEDDINGS = 'WordEmbeddings'
FOLDER_VOCABULARY = 'Vocabulary'
FOLDER_TEXT_CORPUSES = 'TextCorpuses'
FOLDER_UFSAC = 'ufsac-public-2.1'

BN_WORD_INTROS = 'BabelNet_word_intros'
BN_SYNSET_DATA = 'BabelNet_synset_data'
BN_SYNSET_EDGES = 'BabelNet_synset_edges'

FOLDER_WT103 = os.path.join('TextCorpuses','WikiText-103')
FOLDER_WT2 = os.path.join('TextCorpuses','wikitext-2')

WT_TRAIN_FILE = 'wiki.train.tokens'
WT_VALID_FILE = 'wiki.valid.tokens'
WT_TEST_FILE = 'wiki.test.tokens'

PHRASES_MODEL_FILE = 'phrases_model.pickle'
PHRASED_TRAINING_CORPUS = 'phrased_training_corpus.txt'
TEMPORARY_PHRASED_CORPUS = 'temp_phrased_training_corpus.txt'

VOCAB_WT2_FILE = 'vocabulary_from_WikiText-2.h5'
VOCAB_WT103_FILE = 'vocabulary_from_WikiText-103.h5'
VOCAB_FROMSLC_FILE = 'vocabulary_from_SenseLabeledCorpus.h5'
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

ENCODER_MODEL = "Encoder_768to300_nn.model" # not used in the current version


