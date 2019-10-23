import urllib.request
import json
import logging
from enum import Enum
import Filesystem as F
import os
import pandas as pd


class RequestType(Enum):
    WORD_SYNSETS_INTROS = 'WORD_SYNSETS_INTROS'
    SYNSET_DATA = 'SYNSET_DATA'
    SYNSET_EDGES = 'SYNSET_EDGES'


class BabelNetRequestSender:

    def __init__(self):
        self.requests_threshold = 100 # 4500, based on the available amount of BabelCoins (for me currently 5000)
        self.requests_counter = 0 # It is set when we create this object

    # Enty function: If we already sent this request to BabelNet, then it has been stored in one of the hdf5 archives.
    # Search for it, and if we find it we load it.
    # If it is new, then we must proceed with a HTTP request to BabelNet, and save the result
    def lookup_resource(self, req_type, key, target_word='', synset_id=''):
        # choose archive to extract from:
        if req_type == RequestType.WORD_SYNSETS_INTROS:
            archive_fpath = os.path.join(F.FOLDER_INPUT, F.BN_WORD_INTROS)
            babelNet_archive = pd.HDFStore(archive_fpath, mode='r')
            babelNet_archive.select(key=req_type, where="word == '" + target_word + "'")
        elif req_type == RequestType.SYNSET_DATA:
            archive_fpath = os.path.join(F.FOLDER_INPUT, F.BN_SYNSET_DATA)
            babelNet_archive = pd.HDFStore(archive_fpath, mode='r')
            babelNet_archive.select(key=req_type, where="synset_id == '" + synset_id + "'")
        else:
            archive_fpath = os.path.join(F.FOLDER_INPUT, F.BN_SYNSET_EDGES)
            babelNet_archive = pd.HDFStore(archive_fpath, mode='r')
            babelNet_archive.select(key=req_type, where="synset_id == '" + synset_id + "'")


    ########## HTTP Requests

    # For a word, retrieve the "introductory structures" of its synsets, containing id and PoS
    # e.g.: [{'id': 'bn:00030151n', 'pos': 'NOUN', 'source': 'BABELNET'}, {...}, ...]
    def get_syns_intros_word(self, key, target_word, searchLang='EN'):
        req_url = 'https://babelnet.io/v5/getSynsetIds?lemma='+target_word+'&searchLang='+searchLang+'&key='+key
        with urllib.request.urlopen(req_url) as response:
            synsets_intros = json.load(response)
        self.requests_counter = self.requests_counter +1
        logging.debug("BabelNet request: get_syns_intros_word for:" + target_word)
        return synsets_intros

    # Retrieve the data structure of the synset with the given id.
    # It contains everything: glosses, examples, etc.
    def get_synset_data(self, key, synset_ID):
        req_url = 'https://babelnet.io/v5/getSynset?id='+synset_ID+'&key='+key
        with urllib.request.urlopen(req_url) as response:
            synset_data = json.load(response)
        self.requests_counter = self.requests_counter +1
        logging.debug("BabelNet request: get_synset_data for:" + synset_ID)
        return synset_data

    # For a synset, retrieve the relation edges (hypernyms, hyponyms, antonyms etc.)
    def get_synset_edges(self, key, synset_ID):
        req_url = 'https://babelnet.io/v5/getOutgoingEdges?id='+ synset_ID +'&key='+ key
        with urllib.request.urlopen(req_url) as response:
            synset_edges = json.load(response)
        self.requests_counter = self.requests_counter +1
        logging.debug("BabelNet request: get_synset_edges for:" + synset_ID)
        return synset_edges


    ####################

