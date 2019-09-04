import Utils
import urllib.request
import logging
import json
import pandas as pd
import os
from itertools import cycle


########## HTTP Requests

# For a word, retrieve the "introductory structures" of its synsets, containing id and PoS
# e.g.: [{'id': 'bn:00030151n', 'pos': 'NOUN', 'source': 'BABELNET'}, {...}, ...]
def get_syns_intros_word(key, target_word, searchLang='EN'):
    req_url = 'https://babelnet.io/v5/getSynsetIds?lemma='+target_word+'&searchLang='+searchLang+'&key='+key
    with urllib.request.urlopen(req_url) as response:
        synsets_intros = json.load(response)
    return synsets_intros

# Retrieve the data structure of the synset with the given id.
# It contains everything: glosses, examples, etc.
def get_synset_data(key, synset_ID):
    req_url = 'https://babelnet.io/v5/getSynset?id='+synset_ID+'&key='+key
    with urllib.request.urlopen(req_url) as response:
        synset_data = json.load(response)
    return synset_data

####################


# Keep only the relevant synsets for the target word.
# Directives:
# Exclude the synsets where the synsetType is NAMED_ENTITIES instead of CONCEPTS
# Policy: Restrict to WordNet. If there isn’t any WordNetSense in the list of senses, drop
def check_include_synset(target_word, synset_data):
    if synset_data['synsetType'] != 'CONCEPT':
        logging.debug("Named Entity. Ignoring.")
        return False #we do not deal with Named Entities here

    senses = synset_data['senses']

    if not (any([True if (sense['type']=='WordNetSense') else False for sense in senses])):
        logging.debug("No WordNet senses. Ignoring.")
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

####################


def retrieve_DES(target_word='light'):
    Utils.init_logging(os.path.join("GetInputData", "BabelNet.log"), logging.INFO)

    key = Utils.BABELNET_KEY
    accepted_sources = ['WIKI', 'WIKIDIS', 'OMWIKI', 'WN']
    definitions_dict = {} # the key is the synset id
    examples_dict = {}
    synonyms_dict = {}
    syns_intros = get_syns_intros_word(key, target_word)
    synset_ids = list(map(lambda syns_intro_dict: syns_intro_dict["id"], syns_intros))
    logging.debug(synset_ids)

    for s_id in synset_ids:
        synset_data = get_synset_data(key, s_id)
        if check_include_synset(target_word, synset_data):
            logging.info("Including synset for:" + str(s_id))
            definitions = extract_definitions(synset_data,accepted_sources)
            if len(definitions) < 1: #No definitions were found from approved sources.
                logging.info("No definitions from the approved sources were found. Excluding synset")
                continue
            else:
                definitions_dict[s_id] = definitions
                logging.info(definitions)

                examples = extract_examples(synset_data)
                if len(examples) > 1:
                    examples_dict[s_id] = examples
                logging.info(examples)

                synonyms = extract_synonyms(synset_data)
                synonyms = list(filter(lambda sy: sy != target_word, synonyms))
                if len(synonyms) > 1:
                    synonyms_dict[s_id] = synonyms
                logging.info(synonyms)

    logging.info(definitions_dict)
    logging.info(examples_dict)
    logging.info(synonyms_dict)

    return definitions_dict, examples_dict, synonyms_dict


