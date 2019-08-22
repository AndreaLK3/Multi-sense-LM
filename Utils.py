import logging
import sys
import pandas as pd
import langid
import nltk
import string

# Constants
BABELNET_KEY = '7ba5e9a1-1f42-4d9a-97a7-c888975a60a1' #1000 queries per day. Wrote e-mail to request 5000
THESAURUS_KEY = 'xIOLizqsIEV9CMmfkza9' #upto 5000 queries per day.

SOURCE_WORDNET = 'WordNet'
SOURCE_BABELNET = 'BabelNet'
SOURCE_WIKTIONARY = 'Wiktionary'
SOURCE_OMEGAWIKI = 'OmegaWiki'
SOURCE_DBPEDIA = "DBpedia"

DEFINITIONS = 'definitions'
EXAMPLES = 'examples'
SYNONYMS = 'synonyms'
ANTONYMS = 'antonyms'
ENCYCLOPEDIA_DEF = 'encyclopedia_def'
PREP_DEFINITIONS = 'preprocessed_definitions'
PREP_EXAMPLES = 'preprocessed_examples'

FOLDER_INPUT = 'InputData'

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


def read_hdf5_storage(filepath):
    df = pd.read_hdf(filepath)
    return df



### Note: must add the vocabularies of other languages
def check_language(text, lang_id):

    languages_lts = langid.rank(text)
    possible_match = (lang_id.lower() in [lang_tuple[0] for lang_tuple in languages_lts[0:5]])

    text_tokens = nltk.tokenize.word_tokenize(text.lower())
    text_tokens_nopunct = list(filter(lambda t: t not in string.punctuation, text_tokens))

    if not(possible_match):
        logging.warning("Examining vocabulary for element : '" + str(text) +"'")
        if all([t in nltk.corpus.words.words() for t in text_tokens_nopunct]):
            possible_match = True #all the words can be found in the vocabulary of the target language

    if not possible_match:
        logging.warning("Not of language : " + lang_id + " Element : '" + str(text) + "'")
    return possible_match


HDF5_BASE_CHARSIZE = 1024