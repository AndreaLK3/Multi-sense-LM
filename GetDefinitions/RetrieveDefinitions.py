import Utils
import logging
import WordNet
import BabelNet
import DBpedia
import Wiktionary
import OmegaWiki
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

NUM_WORDS_IN_FILE = 5000



def main():
    Utils.init_logging("RetrieveDefinitions.log", logging.INFO)

    with pd.HDFStore('Desa_OmegaWiki.h5', mode='w') as d: #reset HDF5 archives
        pass

    toy_vocabulary = ['high'] #toy_vocabulary = ["sunlight", "plant", "sea", "high", "move"]

    # Note: the words of the vocabulary must be processed in alphabetic order, to guarantee that the data in the
    # tables in the HDF5 storage is sorted.
    toy_vocabulary_sorted = sorted(toy_vocabulary)

    tasks = [WordNet.retrieve_DESA,  BabelNet.retrieve_DES,
             Wiktionary.retrieve_DE, OmegaWiki.retrieve_DS, DBpedia.retrieve_dbpedia_def]
    sources = [Utils.SOURCE_WORDNET, Utils.SOURCE_BABELNET, Utils.SOURCE_WIKTIONARY, Utils.SOURCE_OMEGAWIKI, Utils.SOURCE_DBPEDIA]
    columns_to_update = [range(4), range(3), range(2), [0,2], [4]]
    categories = [Utils.DEFINITIONS,  Utils.EXAMPLES, Utils.SYNONYMS, Utils.ANTONYMS, Utils.ENCYCLOPEDIA_DEF]

    # WordNet
    for i in range(len(tasks)):
        task = tasks[i]
        source = sources[i]
        to_update = columns_to_update[i]

    with pd.HDFStore('Desae.h5', mode='a') as storage:
        for word in toy_vocabulary_sorted:
            desa = task(word)
            for j in to_update:
                df = pd.DataFrame(data=desa[j], columns=[categories[j]])
                df.to_hdf(storage, key=categories[j])


