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


def read_desae(category_name):
    with pd.HDFStore('Desae.h5', mode='r') as storage:
        df = pd.read_hdf(storage, key=category_name)

    return df


def main():
    Utils.init_logging("RetrieveDefinitions.log", logging.INFO)

    with pd.HDFStore('Desa_OmegaWiki.h5', mode='w') as d: #reset HDF5 archives
        pass

    toy_vocabulary = ['high'] #toy_vocabulary = ["sunlight", "plant", "sea", "high", "move"]

    # Note: the words of the vocabulary must be processed in alphabetic order, to guarantee that the data in the
    # tables in the HDF5 storage is sorted.
    toy_vocabulary_sorted = sorted(toy_vocabulary)

    tasks = [WordNet.retrieve_DESA, Wiktionary.retrieve_DE, OmegaWiki.retrieve_DS,
             DBpedia.retrieve_dbpedia_def] #BabelNet.retrieve_DES,
    sources = [Utils.SOURCE_WORDNET, Utils.SOURCE_WIKTIONARY, Utils.SOURCE_OMEGAWIKI, Utils.SOURCE_DBPEDIA] #Utils.SOURCE_BABELNET

    categories = [Utils.DEFINITIONS, Utils.EXAMPLES, Utils.SYNONYMS, Utils.ANTONYMS, Utils.ENCYCLOPEDIA_DEF]
    storage_filenames = [Utils.H5_raw_defs, Utils.H5_examples, Utils.H5_synonyms, Utils.H5_antonyms, Utils.H5_enc_defs]
    open_storage_files = [ pd.HDFStore(fname, mode='a') for fname in storage_filenames]

    categories_returned = [[0,1,2,3],[0,1],[0,2],[4]] #[0,1,2]
    num_columns = [4,2,2,1] #3

    # WordNet
    for i in range(len(tasks)):
        task = tasks[i]
        source = sources[i]
        cats = categories_returned[i]
        n_cols = num_columns[i]
        logging.info(source)

        for word in toy_vocabulary_sorted:
            desa = task(word)
            logging.info(desa)
            for j in range(n_cols):
                logging.info(cats[j])
                df = pd.DataFrame(data=desa[j], columns=[categories[cats[j]]])
                df.to_hdf(open_storage_files[cats[j]], key=categories[cats[j]])

    for storagefile in open_storage_files:
        storagefile.close()

