import Utils
import urllib.request
import logging
import json
import requests


def get_BabelNet_version(key):
    # If you do not pass the data argument, urllib uses a GET request.
    # Data can also be passed in an HTTP GET request by encoding it in the URL itself.
    req_url = 'https://babelnet.io/v5/getVersion?key={' + key +  '}'
    with urllib.request.urlopen(req_url) as response:
        logging.info(response)
        logging.info(response.status)

        version = json.load(response)
        logging.info(version)
    return version


def get_syns_intros_for_word(key, target_word, searchLang='EN'):
    req_url = 'https://babelnet.io/v5/getSynsetIds?lemma='+target_word+'&searchLang='+searchLang+'&key='+key
    with urllib.request.urlopen(req_url) as response:
        synsets_intros = json.load(response)
    return synsets_intros


def get_synset_data(key, synset_ID):
    req_url = 'https://babelnet.io/v5/getSynset?id='+synset_ID+'&key='+key
    with urllib.request.urlopen(req_url) as response:
        synset_info = json.load(response)
        #logging.info(synset_info)
    return synset_info

# Directives:
# Exclude the synsets where the synsetType is NAMED_ENTITIES instead of CONCEPTS
# Policy: Restrict to WordNet. in the list of senses:
# 	- if there isn’t any WordNetSense, drop
#   - go into properties > fullLemma. If the original target word is not contained in any of the lemmas, then drop.
# Collect the definitions: glosses > gloss , and the source must be either WN or WIKI
def extract_definitions_and_sources(synset_data, target_word):

    if synset_data['synsetType'] != 'CONCEPT':
        logging.info("Named Entity. Ignoring.")
        return [] #we do not deal with Named Entities for now

    senses = synset_data['senses']

    if not (any([True if (sense['type']=='WordNetSense') else False for sense in senses])):
        logging.info("No WordNet senses. Ignoring.")
        return []

    lemmas = [sense['properties']['simpleLemma'] for sense in senses]
    logging.info(lemmas)

    if not(any([True if (target_word.lower() == lemma.lower()) else False for lemma in lemmas])):
        logging.info("Target word not found inside lemmas. Ignoring.")
        return []

    accepted_definition_entries = \
        list(filter(lambda defDict: (defDict['source'] == 'WN') or (defDict['source'] == 'WIKI'), synset_data['glosses']))

    defs_and_sources = list(map( lambda defDict : (defDict['gloss'],defDict['source']) ,accepted_definition_entries))

    return defs_and_sources


def main():
    Utils.init_logging("defs_BabelNet.log", logging.INFO)
    key = '7ba5e9a1-1f42-4d9a-97a7-c888975a60a1'

    target_word = 'plant'

    #get_BabelNet_version(key)
    syns_intro_dicts = get_syns_intros_for_word(key, target_word)
    logging.info(syns_intro_dicts)

    synset_ids = list( map( lambda syns_intro_dict : syns_intro_dict["id"], syns_intro_dicts))

    for s_id in synset_ids:
        logging.info(s_id)
        synset_data = get_synset_data(key, s_id)
        def_source_lts = extract_definitions_and_sources(synset_data, target_word)
        logging.info(def_source_lts)

main()