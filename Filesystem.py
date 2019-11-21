import os

FOLDER_INPUT = 'InputData'
FOLDER_WORD_EMBEDDINGS = 'WordEmbeddings'
FOLDER_VOCABULARY = 'Vocabulary'
FOLDER_TEXT_CORPUSES = 'TextCorpuses'
FOLDER_SENSEANNOTATED = 'SenseAnnotated'
FOLDER_GOOGLE_MASC_SEMCOR = 'Google_Masc&SemCor'

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
VOCAB_PHRASED = 'vocabulary_phrased.h5'

VOCAB_CURRENT_INDEX_FILE = 'vocabulary_currentIndex.txt' # used for BabelNet requests over several days

FASTTEXT_PRETRAINED_EMBEDDINGS_FILE = 'cc.en.300.bin'
SPVs_FASTTEXT_FILE = 'SinglePrototypes_withFastText.npy'
SPVs_DISTILBERT_FILE = 'SinglePrototypes_withDistilBERT.npy'

FOLDER_MASC = 'masc'
FOLDER_MASC_WRITTEN = 'written'
FOLDER_SEMCOR = 'semcor'

NOAD_WORDNET_AUTOMAP_FILE = 'algorithmic_map.txt'
NOAD_WORDNET_MANUALMAP_FILE = 'manual_map.txt'
SEMCOR_DB = 'semcor_all.db'
MASC_DB = 'masc_written.db'

ENCODER_MODEL = "Encoder_768to300_nn.model" # not used in the current version


