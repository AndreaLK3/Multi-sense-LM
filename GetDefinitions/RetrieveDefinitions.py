import Utils
import logging
import WordNet
import BabelNet
import Wiktionary
import OmegaWiki

import pandas



def main():
    Utils.init_logging("RetrieveDefinitions.log", logging.INFO)

    babelnet_key = '7ba5e9a1-1f42-4d9a-97a7-c888975a60a1'
    toy_vocabulary = ["plant", "sea", "high", "move"]

    # Note: the words of the vocabulary must be processed in alphabetic order, to guarantee that the data in the
    # tables in the HDF5 storage is sorted.

    toy_vocabulary_sorted = sorted(toy_vocabulary)


    tasks = [WordNet.process_all_synsets_of_word, BabelNet.get_defs_sources_word,
             Wiktionary.get_defs_and_examples, OmegaWiki.get_definitions]


