import logging
import sys
import langid
import nltk
import string
import re
import time
import torch
import os
import sqlite3

########## Constants ##########

HDF5_BASE_SIZE_512 = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu')#

# Lexicon

STANDARDTEXT = 'StandardText'
SENSELABELED = 'SenseLabeled'

DEFINITIONS = 'definitions'
EXAMPLES = 'examples'
SYNONYMS = 'synonyms'
ANTONYMS = 'antonyms'
CATEGORIES = [DEFINITIONS, EXAMPLES, SYNONYMS, ANTONYMS] # , Utils.ENCYCLOPEDIA_DEF

SENSE_WN_ID = 'sense_wn_id'

DENOMINATED = 'denominated'
PROCESSED = 'processed' # for defs and examples, it means: 'no duplicates'; for synonyms and antonyms: 'lemmatized'
VECTORIZED = 'vectorized'
DISTILBERT = "DistilBERT"
FASTTEXT = "FastText"
TXL = "TXL"

TRAINING = 'Training'
VALIDATION = 'Validation'
TEST = 'Test'

INDICES_TABLE_DB = 'indices_table.sql'

UNK_TOKEN = '<unk>'
NUM_TOKEN = '<num>'
EOS_TOKEN = '<eos>'

EMPTY = 'EMPTY'
MOST_FREQUENT_SENSE = "MostFrequentSense"
WORD = 'word'
INDEX = 'index'

CORRECT_PREDICTIONS = 'correct_predictions'

GRAPH_EMBEDDINGS_DIM = 300

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

def get_timestamp_month_to_sec():
    return '_'.join([str(time.localtime().tm_mon), str(time.localtime().tm_mday), str(time.localtime().tm_hour),
              str(time.localtime().tm_min), str(time.localtime().tm_sec)])

def time_measurement_with_msg(t0, t1, message):
    logging.info(message + ". Time elapsed=" + str(round(t1 - t0, 3)) + " s")

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


def word_to_vocab_index(word, vocabulary_wordList):
    try:
        return vocabulary_wordList.index(word)
    except ValueError:
        return vocabulary_wordList.index(UNK_TOKEN)


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

# Read the indices_table.sql, in order to determine the start of the dummmySenses.
def get_startpoint_dummySenses(inputdata_folder):
    counter = 0

    db_filepath = os.path.join(inputdata_folder, INDICES_TABLE_DB)
    indicesTable_db = sqlite3.connect(db_filepath)
    indicesTable_db_c = indicesTable_db.cursor()
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

# this can be used to compute the startpoint of the dummySenses, by executing |senses| - (|senses| - |definitions|)
def compute_startpoint_dummySenses(graph_dataobj):
    len_senses = graph_dataobj.node_types.tolist().index(1)
    len_defs = graph_dataobj.node_types.tolist().index(3) - graph_dataobj.node_types.tolist().index(2)
    num_dummySenses = len_senses - len_defs
    startpoint_dummySenses = len_senses - num_dummySenses
    return startpoint_dummySenses


