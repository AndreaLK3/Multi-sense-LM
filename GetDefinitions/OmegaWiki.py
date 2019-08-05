import Utils
import urllib.request
import json
import logging



def get_data_structure(target_word):
    req_url = 'http://www.omegawiki.org/api.php?action=ow_express&search=' + target_word + '&format=json'
    with urllib.request.urlopen(req_url) as response:
        structure = json.load(response)
    return structure

def get_definitions(structure):
    definitions = []

    def_dict_keys = (structure['ow_express']).keys()
    defs_keys = list(filter(lambda key: key.startswith('ow_define'), def_dict_keys))

    for key in defs_keys:
        if (structure['ow_express'][key]['lang'] == 'English'):
            definitions.append(structure['ow_express'][key]['definition']['text'])

    return definitions

# I still need the data structure with the Definition Meanings, since the synonyms are connected
# to those (and not to the "abstract, all-encompassing" target word
def get_synonyms(structure):
    synonyms = []

    def_dict_keys = (structure['ow_express']).keys()
    defs_keys = list(filter(lambda key: key.startswith('ow_define'), def_dict_keys))

    defs_keys_EN = list(filter( lambda key: structure['ow_express'][key]['lang']=='English', defs_keys))

    dm_ids = list(map(lambda key: structure['ow_express'][key]['dmid'] ,defs_keys_EN))

    #n: with English, lang=85
    for dm_id in dm_ids:
        req_url = 'http://www.omegawiki.org/api.php?action=ow_syntrans&dm='+str(dm_id)+'&part=syn&lang=85&format=json'
        with urllib.request.urlopen(req_url) as response:
            syn_structure = json.load(response)['ow_syntrans']
            num_synonyms = len(syn_structure)-2
            for i in range (1, num_synonyms+1):
                synonym_key = str(i)+'.'
                synonyms.append(syn_structure[synonym_key]['e'])

    return synonyms


def retrieve_DS(target_word):
    #Utils.init_logging("OmegaWiki.log", logging.INFO)

    st = get_data_structure(target_word)
    defs = get_definitions(st)

    synonyms = get_synonyms(st)
    synonyms = list(dict.fromkeys(synonyms))

    return defs, synonyms



#main()