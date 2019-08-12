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
HDF5_BASE_CHARSIZE = 512

def main():
    Utils.init_logging("RetrieveInputData.log", logging.INFO)

    toy_vocabulary = ['high'] #TODO: check language and completeness of this example
    #toy_vocabulary = ["sunlight", "plant", "sea", "high", "move"]

    # Note: the words of the vocabulary must be processed in alphabetic order, to guarantee that the data in the
    # tables in the HDF5 storage is sorted.
    toy_vocabulary_sorted = sorted(toy_vocabulary)

    tasks = [WordNet.retrieve_DESA, Wiktionary.retrieve_DE, OmegaWiki.retrieve_DS,
             DBpedia.retrieve_dbpedia_def, BabelNet.retrieve_DES]
    sources = [Utils.SOURCE_WORDNET, Utils.SOURCE_WIKTIONARY, Utils.SOURCE_OMEGAWIKI, Utils.SOURCE_DBPEDIA, Utils.SOURCE_BABELNET]

    categories = [Utils.DEFINITIONS, Utils.EXAMPLES, Utils.SYNONYMS, Utils.ANTONYMS, Utils.ENCYCLOPEDIA_DEF]
    hdf5_min_itemsizes_dict = {Utils.DEFINITIONS:HDF5_BASE_CHARSIZE, Utils.EXAMPLES:2*HDF5_BASE_CHARSIZE,
                               Utils.SYNONYMS:HDF5_BASE_CHARSIZE, Utils.ANTONYMS:HDF5_BASE_CHARSIZE,
                               Utils.ENCYCLOPEDIA_DEF:4*HDF5_BASE_CHARSIZE}

    storage_filenames = [Utils.H5_raw_defs, Utils.H5_examples, Utils.H5_synonyms, Utils.H5_antonyms, Utils.H5_enc_defs]
    open_storage_files = [ pd.HDFStore(fname, mode='w') for fname in storage_filenames] #reset HDF5 archives

    categories_returned = [[0,1,2,3],[0,1],[0,2],[4], [0,1,2]]
    num_columns = [4,2,2,1, 3]

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

                (open_storage_files[cats[j]]).append(key=categories[cats[j]], value=df,
                                                     min_itemsize={key:hdf5_min_itemsizes_dict[key] for key in [categories[cats[j]]]})

    for storagefile in open_storage_files:
        storagefile.close()

