import logging
import sys
import langid
import nltk
import string
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from math import exp
import torch
import os
import Filesystem as F
import sqlite3

########## Constants ##########

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

GLOBALS = 'globals'
SENSES = 'senses'
CORRECT_PREDICTIONS = 'correct_predictions'
TOTAL = 'total'

GRAPH_EMBEDDINGS_DIM = 200

########## Logging and development ##########


########## Text logging ##########

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
    logging.info("*** Chronometer:")
    for i in range(len(time_measurements)-1):
        t1 = time_measurements[i]
        t2 = time_measurements[i+1]
        logging.info('t'+str(i+1)+' - t'+str(i)+' = '+str(round(t2-t1,5)))

def get_timestamp_month_to_min():
    return '_'.join([str(time.localtime().tm_mon), str(time.localtime().tm_mday), str(time.localtime().tm_hour),
              str(time.localtime().tm_min)])


def record_statistics(epoch_sumlosses_tpl, epoch_numsteps_tpl, losses_lts):
    sum_epoch_loss_global,sum_epoch_loss_sense, sum_epoch_loss_multisense = epoch_sumlosses_tpl
    epoch_step, num_steps_withsense, num_steps_withmultisense = epoch_numsteps_tpl
    if num_steps_withsense==0: num_steps_withsense=1 # adjusting for when we do only standard LM
    if num_steps_withmultisense == 0:  num_steps_withmultisense = 1

    epoch_loss_globals = sum_epoch_loss_global / epoch_step
    epoch_loss_senses = sum_epoch_loss_sense / num_steps_withsense
    epoch_loss_multisenses = sum_epoch_loss_multisense / num_steps_withmultisense

    epoch_loss = epoch_loss_globals + epoch_loss_senses
    logging.info("Losses: " + " Globals loss=" + str(round(epoch_loss_globals,2)) +
                               " \tSense loss=" + str(round(epoch_loss_senses,2)) +
                               " \tLoss on multi-senses=" + str(round(epoch_loss_senses, 2)) +
                               " \tTotal loss=" + str(round(epoch_loss,3)) )
    logging.info("Perplexity: " + " Globals perplexity=" + str(round(exp(epoch_loss_globals),2)) +
                 " \tPerplexity on all senses=" + str(round(exp(epoch_loss_senses),2)) +
                 " \tPerplexity on multi-senses=" + str(round(exp(epoch_loss_multisenses),2)) + "\n-------")
    losses_lts.append((epoch_loss_globals, epoch_loss_senses))

##########

########## Graphics logging ##########

def display_ygraph_from_nparray(data_y_array, axis_labels=None, label=None):

    # data_y_array = np.load(npy_fpath, allow_pickle=True)
    plt.plot(data_y_array, label=label, marker='.')
    plt.xticks(range(0, int(len(data_y_array)) + 1, max(int(len(data_y_array))//20, 1)))##np.arange(len(data_y_array)), np.arange(1, len(data_y_array)+1))
    plt.yticks(range(0, int(max(data_y_array)) + 1, max(int(max(data_y_array))//20, 1)))
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




##########

########## Other utilities ##########

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


### When we encounter UNK, reading in text for the NN, we skip it
class MustSkipUNK_Exception(Exception):
    def __init__(self):
        super().__init__()


# Utility for processing entities, word embeddings & co
def count_tokens_in_corpus(corpus_txt_filepath, include_punctuation):

    file = open(corpus_txt_filepath, "r") # encoding="utf-8"
    tot_tokens = 0

    for i, line in enumerate(file):
        if line == '':
            break
        # tokens_in_line = nltk.tokenize.word_tokenize(line)
        line_noPuncts = re.sub('['+string.punctuation.replace('-', '')+']', ' ', line)
        if not (include_punctuation):
            the_line = line_noPuncts
        else:
            the_line = line
        tokens_in_line = nltk.tokenize.word_tokenize(the_line)
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

### Selecting from a HDF5 archive, and dealing with the possible syntax errors
### e.g.: where word == and, or ==''s ' or =='\'
def select_from_hdf5(input_db, table_key, field_names, values):
    #word_df = input_db.select(key=elements_name, where="word == '" + str(word) + "'")
    logging.debug("values (parameter)=" + str(values))
    values = list(map(lambda v: v.replace("'", ""), values))
    values = list(filter(lambda v:'\\' not in v, values))
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
    try:
        word = sense_str[0:get_locations_of_char(sense_str, '.')[-2]]
    except AttributeError:
        logging.debug("sense_str= " + sense_str)
        pattern_2 = '.[^.]+'
        mtc = re.match(pattern_2, sense_str)
        word = mtc.group() if mtc is not None else '.'
    return word


# Returns a list of indices: where the character is located in the word/string.
# Useful to split senses, e.g. abbreviate.v.02 or Sr..dummySense.01
def get_locations_of_char(word, char):
    locations = []
    for i in range(len(word)):
        c = word[i]
        if c == char:
            locations.append(i)
    return locations



##### Check GPU memory usage
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


# Read the indices_table.sql, in order to determine the start of the dummmySenses.
def get_startpoint_dummySenses():
    indicesTable_db = sqlite3.connect(os.path.join(F.FOLDER_INPUT, INDICES_TABLE_DB))
    indicesTable_db_c = indicesTable_db.cursor()
    counter = 1

    indicesTable_db_c.execute("SELECT * FROM indices_table")
    while (True):
        db_row = indicesTable_db_c.fetchone()
        if db_row is None:
            break
        sense_id = db_row[0]
        if 'dummySense' in sense_id:
            return counter
        counter = counter +1
    return counter
