import urllib.request
import json
import logging

class BabelNetRequestSender:

    def __init__(self):
        self.requests_threshold = 200 # 4500, based on the available amount of BabelCoins (for me currently 5000)
        self.requests_counter = 0 # It is set when we create this object

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

