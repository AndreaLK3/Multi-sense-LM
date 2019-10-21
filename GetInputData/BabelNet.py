import Utils
import urllib.request
import logging
import json
import pandas as pd
import os
from itertools import cycle


class BabelNet_RequestSender:

    def __init__(self):
        self.requests_threshold = 20 # 4900 based on the available amount of BabelCoins (for me currently 5000)
        self.requests_counter = 0 # It is set when we create this object

    ########## HTTP Requests

    # For a word, retrieve the "introductory structures" of its synsets, containing id and PoS
    # e.g.: [{'id': 'bn:00030151n', 'pos': 'NOUN', 'source': 'BABELNET'}, {...}, ...]
    def get_syns_intros_word(self, key, target_word, searchLang='EN'):
        req_url = 'https://babelnet.io/v5/getSynsetIds?lemma='+target_word+'&searchLang='+searchLang+'&key='+key
        with urllib.request.urlopen(req_url) as response:
            synsets_intros = json.load(response)
        self.requests_counter = self.requests_counter =+1
        return synsets_intros

    # Retrieve the data structure of the synset with the given id.
    # It contains everything: glosses, examples, etc.
    def get_synset_data(self, key, synset_ID):
        req_url = 'https://babelnet.io/v5/getSynset?id='+synset_ID+'&key='+key
        with urllib.request.urlopen(req_url) as response:
            synset_data = json.load(response)
        self.requests_counter = self.requests_counter = +1
        return synset_data

    # For a synset, retrieve the relation edges (hypernyms, hyponyms, antonyms etc.)
    def get_synset_edges(self, key, synset_ID):
        req_url = 'https://babelnet.io/v5/getOutgoingEdges?id='+ synset_ID +'&key='+ key
        with urllib.request.urlopen(req_url) as response:
            synset_edges = json.load(response)
        self.requests_counter = self.requests_counter = +1
        return synset_edges


    ####################


# Keep only the relevant synsets for the target word.
# Directives:
# Exclude the synsets where the synsetType is NAMED_ENTITIES instead of CONCEPTS
# Policy: Restrict to WordNet. If there isn’t any WordNet definition, drop
# (drop also if the target word is not found among the lemmas of the synset)
def check_include_synset(target_word, synset_data):
    if synset_data['synsetType'] != 'CONCEPT':
        logging.debug("Named Entity. Ignoring.")
        return False #we do not deal with Named Entities here

    defs = synset_data['glosses']

    if not (any([True if (definition['source']=='WN') else False for definition in defs])):
        logging.debug("No WordNet senses. Ignoring.")
        return False

    senses = synset_data['senses']
    lemmas = [sense['properties']['simpleLemma'] for sense in senses if sense['properties']['language']=='EN']

    if not (any([True if (target_word.lower() == lemma.lower()) else False for lemma in lemmas])):
        logging.debug("Target word not found inside lemmas. Ignoring.")
        return False

    return True


#################### Extract elements from the data structure

# Get dictionary definitions.
# Collect the definitions: synset_data > glosses [list of dicts] > gloss
def extract_definitions(synset_data, accepted_sources):

    accepted_definition_entries = list(filter(lambda defDict: defDict['source'] in accepted_sources and defDict['language']=='EN',
                                              synset_data['glosses']))
    defs = list(map( lambda defDict : defDict['gloss'], accepted_definition_entries))
    return defs


def extract_examples(synset_data):
    examples_text = [ex['example'] for ex in synset_data['examples'] if ex['language']=='EN']
    return list(set(examples_text))


def extract_synonyms(synset_data):
    synonyms = []
    senses = synset_data['senses']

    for s in senses:
        properties = s['properties']
        if properties['language'] == 'EN':
            synonyms.append(properties['simpleLemma'])
    #return in lowercase
    return list(map(lambda s: s.lower() ,synonyms))


def extract_antonyms(BN_request_sender, key, synset_edges):

    antonyms_edges = list(filter(lambda edge: edge['pointer']['name'] == 'Antonym', synset_edges))
    antonyms_bnIds = list(map( lambda a_e: a_e['target'] ,antonyms_edges))
    logging.debug("Antonyms: " + str(len(antonyms_bnIds)) + " babelNet IDs")
    antonym_words = []

    for synset_id in antonyms_bnIds:
        antonym_data = BN_request_sender.get_synset_data(key, synset_id)
        antonym_word = antonym_data['senses'][0]['properties']['simpleLemma']
        antonym_words.append(antonym_word)

    return antonym_words


####################


def retrieve_DESA(BN_request_sender, target_word):
    #Utils.init_logging(os.path.join("GetInputData", "BabelNet.log"), logging.INFO)

    key = Utils.BABELNET_KEY
    accepted_sources = ['WIKI', 'WIKIDIS', 'OMWIKI', 'WN']

    definitions_dict = {} # the key is the synset id
    examples_dict = {}
    synonyms_dict = {}
    antonyms_dict = {}

    syns_intros = BN_request_sender.get_syns_intros_word(key, target_word)
    synset_ids = list(map(lambda syns_intro_dict: syns_intro_dict["id"], syns_intros))
    logging.debug(synset_ids)

    for s_id in synset_ids:
        synset_data = BN_request_sender.get_synset_data(key, s_id)
        if check_include_synset(target_word, synset_data):
            logging.debug("Including synset for:" + str(s_id))
            definitions = extract_definitions(synset_data,accepted_sources)
            if len(definitions) < 1: #No definitions were found from approved sources.
                logging.debug("BN: No definitions from the approved sources were found. Excluding synset")
                continue
            else:
                definitions_dict[s_id] = definitions

                examples = extract_examples(synset_data)
                examples_dict[s_id] = examples

                synonyms = extract_synonyms(synset_data)
                synonyms_dict[s_id] = synonyms
                logging.debug("From BabelNet: synonyms : " + str(synonyms))

                synset_edges = BN_request_sender.get_synset_edges(key, s_id)
                antonyms = extract_antonyms(BN_request_sender, key, synset_edges) # extracted from the 'target' of the Antonym edges
                antonyms_dict[s_id] = antonyms
        else:
            logging.debug("Excluding synset for:" + str(s_id))

    return definitions_dict, examples_dict, synonyms_dict, antonyms_dict


