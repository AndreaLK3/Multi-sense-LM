import Utils
import urllib.request
import logging
import json

########## HTTP Requests
def get_syns_intros_word(key, target_word, searchLang='EN'):
    req_url = 'https://babelnet.io/v5/getSynsetIds?lemma='+target_word+'&searchLang='+searchLang+'&key='+key
    with urllib.request.urlopen(req_url) as response:
        synsets_intros = json.load(response)
    return synsets_intros

def get_synset_data(key, synset_ID):
    req_url = 'https://babelnet.io/v5/getSynset?id='+synset_ID+'&key='+key
    with urllib.request.urlopen(req_url) as response:
        synset_data = json.load(response)
    return synset_data
####################



# Keep only the relevant synsets for the target word.
# Directives:
# Exclude the synsets where the synsetType is NAMED_ENTITIES instead of CONCEPTS
# Policy: Restrict to WordNet. in the list of senses:
# 	- if there isn’t any WordNetSense, drop
#   - go into properties > fullLemma. If the original target word is not contained in any of the lemmas, then drop.
def check_include_synset(target_word, synset_data):
    if synset_data['synsetType'] != 'CONCEPT':
        logging.debug("Named Entity. Ignoring.")
        return False #we do not deal with Named Entities here

    senses = synset_data['senses']

    if not (any([True if (sense['type']=='WordNetSense') else False for sense in senses])):
        logging.debug("No WordNet senses. Ignoring.")
        return False

    lemmas = [sense['properties']['simpleLemma'] for sense in senses]

    if not(any([True if (target_word.lower() == lemma.lower()) else False for lemma in lemmas])):
        logging.debug("Target word not found inside lemmas. Ignoring.")
        return False

    return True


# Get dictionary definitions.
# Collect the definitions: glosses > gloss
def extract_definitions(synset_data):

    accepted_definition_entries = list(filter(lambda defDict: defDict['source'] == 'WIKI', synset_data['glosses']))

    defs = list(map( lambda defDict : (defDict['gloss']) ,accepted_definition_entries))
    return defs


def extract_examples(synset_data):
    return [ex['example'] for ex in synset_data['examples']]

def extract_synonyms(synset_data):
    synonyms = []
    senses = synset_data['senses']

    for s in senses:
        synonyms.append(s['properties']['simpleLemma'])
    #return in lowercase
    return list(map(lambda s: s.lower() ,synonyms))

def retrieve_DES(target_word):
    key = Utils.BABELNET_KEY
    definitions = []
    examples = []
    synonyms = []
    syns_intros = get_syns_intros_word(key, target_word)
    synset_ids = list(map(lambda syns_intro_dict: syns_intro_dict["id"], syns_intros))
    logging.debug(synset_ids)

    for s_id in synset_ids:
        synset_data = get_synset_data(key, s_id)
        if check_include_synset(target_word, synset_data):
            definitions.append(extract_definitions(synset_data))
            examples.extend(extract_examples(synset_data))
            synonyms.extend(extract_synonyms(synset_data))


    synonyms = list(dict.fromkeys(synonyms))

    return (definitions, examples, synonyms)




def main():
    Utils.init_logging("defs_BabelNet.log", logging.INFO)
    #key = '7ba5e9a1-1f42-4d9a-97a7-c888975a60a1'

    target_word = 'sea'
    logging.info(retrieve_DES(target_word))



#main()