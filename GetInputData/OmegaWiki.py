import Utils
import urllib.request
import json
import logging


########## HTTP request
def get_data_structure(target_word):
    req_url = 'http://www.omegawiki.org/api.php?action=ow_express&search=' + target_word + '&format=json'
    with urllib.request.urlopen(req_url) as response:
        structure = json.load(response)
    return structure
##########

def get_definitions(structure):
    definitions = []

    def_dict_keys = (structure['ow_express']).keys()
    defs_keys = list(filter(lambda key: key.startswith('ow_define'), def_dict_keys))

    for key in defs_keys:
        if (structure['ow_express'][key]['lang'] == 'English'):
            definitions.append(structure['ow_express'][key]['definition']['text'])

    return definitions



def get_synonyms_for_sense(dm_id):
    synonyms_ls = []
    req_url = 'http://www.omegawiki.org/api.php?action=ow_syntrans&dm=' + str(dm_id) + '&part=syn&lang=85&format=json'
    with urllib.request.urlopen(req_url) as response:
        syn_structure = json.load(response)['ow_syntrans']
        num_synonyms = len(syn_structure) - 2
        for i in range(1, num_synonyms + 1):
            synonym_key = str(i) + '.'
            synonyms_ls.append(syn_structure[synonym_key]['e'])
    return synonyms_ls


# I still need the data structure with the Definition Meanings, since the synonyms are connected
# to those (and not to the "abstract, all-encompassing" target word)
def get_synonyms(structure):
    synonyms = []

    def_dict_keys = (structure['ow_express']).keys()
    defs_keys = list(filter(lambda key: key.startswith('ow_define'), def_dict_keys))

    defs_keys_EN = list(filter( lambda key: structure['ow_express'][key]['lang']=='English', defs_keys))

    dm_ids = list(map(lambda key: structure['ow_express'][key]['dmid'] ,defs_keys_EN))

    #n: with English, lang=85
    for dm_id in dm_ids:
        synonyms.extend(get_synonyms_for_sense(dm_id))
    return synonyms




def lookup_bndefs_dict(ow_define_i, bn_defs_dict):
    ow_def = ow_define_i['definition']['text']

    for key_bn_id in bn_defs_dict.keys():
        if (ow_define_i['lang'] == 'English'):
            bn_defs_ls = bn_defs_dict[key_bn_id]
            if ow_def in bn_defs_ls:
                logging.info("Match found for ow_def='" + str(ow_def) + "' at bn_id=" + str(key_bn_id))
                return key_bn_id
    return None



def retrieve_S(target_word, bn_defs_dict):
    #Utils.init_logging("OmegaWiki.log", logging.INFO)
    synonyms_dict = {}

    ow_data = get_data_structure(target_word)
    ow_dict_keys = (ow_data['ow_express']).keys()
    ow_defs_keys = list(filter(lambda key: key.startswith('ow_define'), ow_dict_keys))

    for ow_key in ow_defs_keys:
        ow_define_i = ow_data['ow_express'][ow_key]
        bn_id = lookup_bndefs_dict(ow_define_i, bn_defs_dict)
        if bn_id is not None:
            dm_id = ow_define_i['dmid']
            synonyms = get_synonyms_for_sense(dm_id)
            synonyms = list(dict.fromkeys(synonyms))
            synonyms_dict[bn_id] = synonyms

    return synonyms_dict



#main()