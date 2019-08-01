import Utils
import urllib.request
import json
import logging



def get_definitions(target_word):
    definitions = []
    req_url = 'http://www.omegawiki.org/api.php?action=ow_express&search='+ target_word +'&format=json'
    with urllib.request.urlopen(req_url) as response:
        structure = json.load(response)

    def_dict_keys = (structure['ow_express']).keys()
    defs_keys = list(filter(lambda key: key.startswith('ow_define'), def_dict_keys))

    for key in defs_keys:
        definitions.append(structure['ow_express'][key]['definition']['text'])

    return definitions



def main():
    Utils.init_logging("OmegaWiki.log", logging.INFO)

    x = get_definitions('plant')
    logging.info(x)

#main()