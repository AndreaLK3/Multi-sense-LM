import logging
import sys
import pandas as pd
import pycld2 as cld2

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

#Filenames & folders
H5_raw_defs = 'raw_defs.h5'
H5_examples = 'examples.h5'
H5_synonyms = 'synonyms.h5'
H5_antonyms = 'antonyms.h5'
H5_enc_defs = 'enc_defs.h5'

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



def check_language(text, lang_id):
    bytes = text.encode('utf-8')
    isReliable, textBytesFound, details = cld2.detect(bytes)
    return lang_id.lower() == details[0][1]