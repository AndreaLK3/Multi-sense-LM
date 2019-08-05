import logging
import sys

# Constants
BABELNET_KEY = '7ba5e9a1-1f42-4d9a-97a7-c888975a60a1'

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