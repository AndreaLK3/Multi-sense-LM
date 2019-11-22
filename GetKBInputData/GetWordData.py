import Utils
import logging
import GetKBInputData.WordNet as WordNet
import GetKBInputData.OmegaWiki as OmegaWiki
import GetKBInputData.BabelNet as BabelNet
import pandas as pd
import os
from itertools import cycle


# # note: we assume that {dict_2.keys} \subsetOf {dict_1.keys}
# def merge_dictionaries_withlists(dict_1, dict_2):
#     merged_dict = {}
#
#     for key in dict_1.keys():
#         ls_1 = dict_1[key]
#         ls_2 = []
#         try:
#             ls_2 = dict_2[key]
#         except KeyError:
#             pass
#         merged_dict[key] = ls_1 + ls_2
#
#     return merged_dict
#
# # transforms a dictonary with K1:[a,b], K2:[] into [(K1,a),(K1,b)]
# def unpack_elemsdict_into_columns(dict):
#     wn_ids = []
#     elems = []
#     for k_bnId, v_elemsLs in dict.items():
#         for i in range(len(v_elemsLs)):
#             wn_ids.append(k_bnId)
#             elems.append(v_elemsLs[i])
#     return wn_ids, elems


def unpack_ls_in_tpls(lts):
    unpacked_lts = []
    for tpl in lts:
        id = tpl[0]
        ls = tpl[1]
        for elem in ls:
            unpacked_lts.append((id, elem))
    return unpacked_lts


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


############### Append to HDF5 tables on disk
def store_data_to_hdf5(data_df, h5_outfiles, h5_itemsizes, lang_id='en'):

    for i in range(len(Utils.CATEGORIES)):
        category = Utils.CATEGORIES[i]
        category_elements_df = data_df.loc[:,[Utils.SENSE_WN_ID, category]]

        sensenames_elements_lts = [tuple(r) for r in category_elements_df.values]
        if len(sensenames_elements_lts) == 0:
            return
        if not(isinstance(sensenames_elements_lts[0][1], str)):
            sensenames_elements_lts = unpack_ls_in_tpls(sensenames_elements_lts)

        # removing trailing whitespace
        sensenames_elements_lts_01 = list(map(lambda tpl: (tpl[0], tpl[1].strip()), sensenames_elements_lts))

        sensenames_elements_lts_02 = list(filter(lambda tpl: Utils.check_language(tpl[1], lang_id), sensenames_elements_lts_01))

        df_columns = [Utils.SENSE_WN_ID, category]
        df = pd.DataFrame(data=sensenames_elements_lts_02, columns=df_columns)
        logging.debug(str(len(sensenames_elements_lts_02)))
        try:
            h5_outfiles[i].append(key=category, value=df, min_itemsize={key: h5_itemsizes[key] for key in df_columns})
        except ValueError as exc:
            logging.info(exc)
            logging.info("Continuing...")


############### Entry point function
def getAndSave_multisense_data(word, open_storage_files, lang_id='en'):

    # prepare storage facilities
    hdf5_min_itemsizes_dict = {Utils.SENSE_WN_ID: Utils.HDF5_BASE_SIZE_512 / 4,
                               Utils.DEFINITIONS: Utils.HDF5_BASE_SIZE_512, Utils.EXAMPLES: Utils.HDF5_BASE_SIZE_512,
                               Utils.SYNONYMS: Utils.HDF5_BASE_SIZE_512 / 4,
                               Utils.ANTONYMS: Utils.HDF5_BASE_SIZE_512 / 4}
                               #Utils.ENCYCLOPEDIA_DEF: 4 * Utils.HDF5_BASE_CHARSIZE}

    word_multisense_data_df = WordNet.retrieve_senses_desa(word)

    store_data_to_hdf5(word_multisense_data_df, open_storage_files, hdf5_min_itemsizes_dict)

