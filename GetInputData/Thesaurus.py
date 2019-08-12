import Utils
import urllib.request
import urllib.parse
import json

# Note: This is NOT Thesaurus.com, and the quality seems lower.
# It is not included in the current version.

BASE_THESAURUS_URL = 'http://thesaurus.altervista.org/thesaurus/v1'



def get_syns_ants(target_word, language='en_US'):
    example = 'http://thesaurus.altervista.org/thesaurus/v1?word=peace&language=en_US&key=test_only&output=xml'
    req_url = BASE_THESAURUS_URL + '?key=' + Utils.THESAURUS_KEY + '&word=' + target_word +\
              '&language=' + language + '&output=json'

    with urllib.request.urlopen(req_url) as response_http:
        response_json = json.load(response_http)
        categ_syns_ls = response_json['response']
        syns_ants_str = '|'.join(list(map(lambda x: x['list']['synonyms'], categ_syns_ls)))

        return syns_ants_str
