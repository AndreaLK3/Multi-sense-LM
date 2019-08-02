import Utils
import logging
import WordNet
import BabelNet
import Wiktionary
import OmegaWiki

import pandas



def main():
    Utils.init_logging("RetrieveDefinitions.log", logging.INFO)

    toy_vocabulary = ["plant", "sea", "high", "move"]

    # Note: the words of the vocabulary must be processed in alphabetic order, to guarantee that the data in the
    # tables in the HDF5 storage is sorted.

