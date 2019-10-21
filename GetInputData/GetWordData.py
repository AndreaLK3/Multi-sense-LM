import Utils
import logging
import GetInputData.WordNet as WordNet
import GetInputData.OmegaWiki as OmegaWiki
import GetInputData.BabelNet as BabelNet
import pandas as pd
import os
from itertools import cycle


# note: we assume that {dict_2.keys} \subsetOf {dict_1.keys}
def merge_dictionaries_withlists(dict_1, dict_2):
    merged_dict = {}

    for key in dict_1.keys():
        ls_1 = dict_1[key]
        ls_2 = []
        try:
            ls_2 = dict_2[key]
        except KeyError:
            pass
        merged_dict[key] = ls_1 + ls_2

    return merged_dict

# transforms a dictonary with K1:[a,b], K2:[] into [(K1,a),(K1,b)]
def unpack_elemsdict_into_columns(dict):
    bn_ids = []
    elems = []
    for k_bnId, v_elemsLs in dict.items():
        for i in range(len(v_elemsLs)):
            bn_ids.append(k_bnId)
            elems.append(v_elemsLs[i])
    return bn_ids, elems


# For all: lowercase, and eliminate duplicates.
# For synonyms and antonyms: removes the target word itself, and eliminates multi-word elements if we so choose
def refine_bnid_elements_dict(target_word, elems_dict, exclude_multiword=False):

    new_elems_dict = {}
    space_characters = ['_', ' ']

    for k in elems_dict.keys():
        values_ls = []
        all_values_lowercased = list(map(lambda s: s.lower(), elems_dict[k]))
        values_set = set(all_values_lowercased)
        try:
            values_set.remove(target_word)
        except KeyError:
            pass

        if exclude_multiword:
            for syn in values_set:
                if not(any([c in syn for c in space_characters])):
                    values_ls.append(syn)
        else:
            values_ls = list(values_set)
        new_elems_dict[k] = values_ls

    return new_elems_dict


########## Use the BabelNet, WordNet and OmegaWiki modules to collect the KB data,
########## then refine it and orgaize it in dictionaries
def retrieve_word_multisense_data(BN_request_sender, target_word):

    bn_dicts = BabelNet.retrieve_DESA(BN_request_sender, target_word)
    logging.debug(bn_dicts)
    wn_dicts = WordNet.retrieve_SA_bySenses(target_word, bn_dicts[0])
    logging.debug(wn_dicts)
    ow_syn_dict = OmegaWiki.retrieve_S(target_word, bn_dicts[0])
    logging.debug(ow_syn_dict)

    # merge dictionaries for D,E,S from the various sources
    all_definitions_dict = bn_dicts[0]
    all_examples_dict = bn_dicts[1]
    all_synonyms_dict = merge_dictionaries_withlists(merge_dictionaries_withlists(bn_dicts[2], wn_dicts[0]), ow_syn_dict)
    all_antonyms_dict = merge_dictionaries_withlists(bn_dicts[3], wn_dicts[1])


    all_definitions_dict = refine_bnid_elements_dict(target_word, all_definitions_dict)
    all_examples_dict = refine_bnid_elements_dict(target_word, all_examples_dict)
    all_synonyms_dict = refine_bnid_elements_dict(target_word, all_synonyms_dict, exclude_multiword=True)
    all_antonyms_dict = refine_bnid_elements_dict(target_word, all_antonyms_dict, exclude_multiword=True)

    return all_definitions_dict, all_examples_dict, all_synonyms_dict, all_antonyms_dict
##########

############### Append to HDF5 tables on disk
def store_data_to_hdf5(word, data_dict, elements_col_name, h5_outfile, h5_itemsizes, lang_id='en'):

    bn_ids, elements = unpack_elemsdict_into_columns(data_dict)
    elements_01 = list(map(lambda s: s.strip(), elements)) # remove trailing whitespace

    df_data = list(zip(cycle([word]), bn_ids, elements_01))
    df_data_lang = list(filter(lambda data_tpl: Utils.check_language(data_tpl[2], lang_id), df_data))

    df_columns = ['word', 'bn_id', elements_col_name]
    df = pd.DataFrame(data=df_data_lang, columns=df_columns)
    h5_outfile.append(key=elements_col_name, value=df, min_itemsize={key: h5_itemsizes[key] for key in df_columns})



############### Entry point function
def getAndSave_multisense_data(word, BN_request_sender, open_storage_files, lang_id='en'):

    # prepare storage facilities
    hdf5_min_itemsizes_dict = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'bn_id': Utils.HDF5_BASE_SIZE_512 / 8,
                               Utils.DEFINITIONS: Utils.HDF5_BASE_SIZE_512, Utils.EXAMPLES: Utils.HDF5_BASE_SIZE_512,
                               Utils.SYNONYMS: Utils.HDF5_BASE_SIZE_512 / 4,
                               Utils.ANTONYMS: Utils.HDF5_BASE_SIZE_512 / 4}
                               #Utils.ENCYCLOPEDIA_DEF: 4 * Utils.HDF5_BASE_CHARSIZE}

    d,e,s,a = retrieve_word_multisense_data(BN_request_sender, word)

    store_data_to_hdf5(word, d, Utils.DEFINITIONS, open_storage_files[0], hdf5_min_itemsizes_dict)
    store_data_to_hdf5(word, e, Utils.EXAMPLES, open_storage_files[1], hdf5_min_itemsizes_dict)

    store_data_to_hdf5(word, s, Utils.SYNONYMS, open_storage_files[2], hdf5_min_itemsizes_dict)
    store_data_to_hdf5(word, a, Utils.ANTONYMS, open_storage_files[3], hdf5_min_itemsizes_dict)

