import requests
import Utils
import logging
import wiktionaryparser as WP

URL = "https://en.wiktionary.org/w/api.php"

def parse_for_word(target_word, session):
    PARAMS = {
        'action': "parse",
        'page': target_word,
        'format': "json"
    }

    R = session.get(url=URL, params=PARAMS)
    data = R.json()
    return data


def main():
    Utils.init_logging('Wiktionary.log', logging.INFO)

    #S = requests.Session()
    target_word = "plant"

    #word_data = parse_for_word(target_word, S)

    parser = WP.WiktionaryParser()
    word_data = parser.fetch(target_word)
    logging.debug(word_data)

    defs_structure = word_data['definitions']

  

main()