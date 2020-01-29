import logging
import sys
import langid
import nltk
import string
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import torch

########## Constants ##########
import torch

BABELNET_KEY = '7ba5e9a1-1f42-4d9a-97a7-c888975a60a1' # 5000 queries per day until 31-12-2019, then 1000

SOURCE_WORDNET = 'WordNet'
SOURCE_BABELNET = 'BabelNet'
SOURCE_WIKTIONARY = 'Wiktionary'
SOURCE_OMEGAWIKI = 'OmegaWiki'
SOURCE_DBPEDIA = "DBpedia"

HDF5_BASE_SIZE_512 = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu')#

# Lexicon

DEFINITIONS = 'definitions'
EXAMPLES = 'examples'
SYNONYMS = 'synonyms'
ANTONYMS = 'antonyms'
ENCYCLOPEDIA_DEF = 'encyclopedia_def'
CATEGORIES = [DEFINITIONS, EXAMPLES, SYNONYMS, ANTONYMS] # , Utils.ENCYCLOPEDIA_DEF

SENSE_WN_ID = 'sense_wn_id'

DENOMINATED = 'denominated'
PROCESSED = 'processed' # for defs and examples, it means: 'no duplicates'; for synonyms and antonyms: 'lemmatized'
VECTORIZED = 'vectorized'
DISTILBERT = "DistilBERT"
FASTTEXT = "FastText"

TRAINING = 'training'
VALIDATION = 'validation'
TEST = 'test'

INDICES_TABLE_DB = 'indices_table.sql'

UNK_TOKEN = '<unk>'
NUM_TOKEN = '<num>'
EOS_TOKEN = '<eos>'

SENSE_NOAD = 'Sense_NOAD'
SENSE_WORDNET = 'Sense_WordNet'
EMPTY = 'EMPTY'

########## Logging and development ##########

def init_logging(logfilename, loglevel=logging.INFO):
  for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
  logging.basicConfig(level=loglevel, filename=logfilename, filemode="w",
                      format='%(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
  # previously using the more verbose format='%(asctime)s -%(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
  # print(logging.getLogger())
  if len(logging.getLogger().handlers) < 2:
      outlog_h = logging.StreamHandler(sys.stdout)
      outlog_h.setLevel(loglevel)
      logging.getLogger().addHandler(outlog_h)


def log_chronometer(time_measurements):
    for i in range(len(time_measurements)-1):
        t1 = time_measurements[i]
        t2 = time_measurements[i+1]
        logging.info('t'+str(i+1)+' - t'+str(i)+' = '+str(round(t2-t1,5)))


def display_ygraph_fromfile(npy_fpath, axis_labels=None):

    data_y_array = np.load(npy_fpath, allow_pickle=True)
    plt.plot(data_y_array)
    plt.xticks(range(0,len(data_y_array), len(data_y_array)//20))
    plt.yticks(range(0, int(max(data_y_array)) + 1, 1))
    plt.xlim((0, len(data_y_array)))
    plt.ylim((0, max(data_y_array)))
    plt.grid(b=True, color='lightgrey', linestyle='-', linewidth=0.5)
    if axis_labels is not None:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])


# For now, intended to be use with training_losses and validation_losses
def display_xygraph_from_files(npy_fpaths_ls):
    overall_max = 0
    legend_labels = ['Training loss', 'Validation loss']
    for i in range(len(npy_fpaths_ls)):
        npy_fpath = npy_fpaths_ls[i]
        xy_lts_array = np.load(npy_fpath, allow_pickle=True)
        plt.plot(xy_lts_array.transpose()[0], xy_lts_array.transpose()[1], label = legend_labels[i])
        array_max = max(xy_lts_array.transpose()[1])
        overall_max = array_max if array_max > overall_max else overall_max
    plt.ylim((0, overall_max))
    ax = plt.axes()
    ax.legend()


#####

### Note: must add the vocabularies of other languages
def check_language(text, lang_id):

    text = text.replace('_', ' ')
    languages_lts = langid.rank(text)
    possible_match = (lang_id.lower() in [lang_tuple[0] for lang_tuple in languages_lts[0:9]])

    text_tokens = nltk.tokenize.word_tokenize(text.lower())
    text_tokens_nopunct = list(filter(lambda t: t not in string.punctuation, text_tokens))

    if not(possible_match):
        logging.debug("Examining vocabulary for element : '" + str(text) +"'")
        if all([t in nltk.corpus.words.words() for t in text_tokens_nopunct]):
            possible_match = True #all the words can be found in the vocabulary of the target language

    if not possible_match:
        logging.info("Not of language : " + lang_id + " Element : '" + str(text) + "'")
    return possible_match


### When we encounter UNK, reading in text for the GNN, we skip it
class MustSkipUNK_Exception(Exception):
    def __init__(self):
        super().__init__()


# Utility for processing entities, word embeddings & co
def count_tokens_in_corpus(corpus_txt_filepath):

    file = open(corpus_txt_filepath, "r") # encoding="utf-8"
    tot_tokens = 0

    for i, line in enumerate(file):
        if line == '':
            break
        # tokens_in_line = nltk.tokenize.word_tokenize(line)
        line_noPuncts = re.sub('['+string.punctuation.replace('-', '')+']', ' ', line)
        tokens_in_line = nltk.tokenize.word_tokenize(line_noPuncts)
        tot_tokens = tot_tokens + len(tokens_in_line)

        if i % 2000 == 0:
            print("Reading in line n. : " + str(i) + ' ; number of tokens encountered: ' + str(tot_tokens))

    file.close()

    return tot_tokens



def word_to_vocab_index(word, vocabulary_wordList):

    try:
        return vocabulary_wordList.index(word)
    except ValueError:
        return vocabulary_wordList.index(UNK_TOKEN)

def close_list_of_files(files_ls):
    for file in files_ls:
        file.close()



# Utility for examining GPU memory usage

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


### Selecting from a HDF5 archive, and dealing with the possible syntax errors
### e.g.: where word == and, or where word ==''s '
def select_from_hdf5(input_db, table_key, field_names, values):
    #word_df = input_db.select(key=elements_name, where="word == '" + str(word) + "'")
    values = list(map(lambda v: v.replace("'", ""), values))
    fields_values_lts = zip(field_names, values)
    query = ""
    for field_value_tpl in fields_values_lts:
        query_part = field_value_tpl[0] + " == '" + field_value_tpl[1] + "'"
        if len(query) == 0:
            query = query_part
        else:
            query = query + " & " + query_part
    df = input_db.select(key=table_key, where=query)
    return df


# e.g.: from 'sea.n.01' get 'sea'
def get_word_from_sense(sense_str):
    pattern = '[^.]+'
    mtc = re.match(pattern, sense_str)
    try:
        word = mtc.group()
    except AttributeError:
        logging.info("sense_str= " + sense_str)
        pattern_2 = '.[^.]+'
        mtc = re.match(pattern_2, sense_str)
        word = mtc.group()
    return word
