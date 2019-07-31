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


def get_syns_intros_for_word(key, target_word='plant', searchLang='EN'):
    req_url = 'https://babelnet.io/v5/getSynsetIds?lemma='+target_word+'&searchLang='+searchLang+'&key='+key
    with urllib.request.urlopen(req_url) as response:
        logging.info(response)
        logging.info(response.status)

        synsets_IDs = json.load(response)
        logging.info(synsets_IDs)
    return synsets_IDs


def get_info_synset(key, synset_ID):
    req_url = 'https://babelnet.io/v5/getSynset?id='+synset_ID+'&key='+key
    with urllib.request.urlopen(req_url) as response:
        logging.info(response)
        logging.info(response.status)

        synset_info = json.load(response)
        logging.info(synset_info)
    return synset_info


def main():
    Utils.init_logging("defs_BabelNet.log", logging.INFO)
    key = '7ba5e9a1-1f42-4d9a-97a7-c888975a60a1'

    #get_BabelNet_version(key)
    syns_intro_dicts = get_syns_intros_for_word(key)

    synset_ids = list( map( lambda syns_intro_dict : syns_intro_dict["id"], syns_intro_dicts))

    for s_id in synset_ids:
        get_info_synset(key, s_id)


main()