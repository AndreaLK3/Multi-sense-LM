import Utils
import os
import pandas as pd
import logging

DEFS_DB_INDEX = 0
EXS_DB_INDEX = 1
SYNS_DB_INDEX = 2
ANTS_DB_INDEX = 3

LOW_SCORE_CUTOFF = 120
HIGH_SCORE_CUTOFF = 135

# The importance score for each sense is based on:
# num of definitions >  num of examples >  num of synonyms > num of antonyms
def compute_importance_score(sense_res_tpl):
    num_defs = sense_res_tpl[0]
    num_exs = sense_res_tpl[1]
    num_syns = sense_res_tpl[2]
    num_ants = sense_res_tpl[3]
    sorting_key = 100 * num_defs + 10 * num_exs + 5 * num_syns + num_ants
    return sorting_key

# Iterate over the list of bn_ids and set the correspondences bn_id-denomination
def get_denominations(sorted_senses_lts):
    denoms_lts = {}
    count_noun = 1; count_verb = 1; count_adj = 1; count_adv = 1
    for tpl in sorted_senses_lts:
        bn_id = tpl[0]
        if bn_id[-1]=='n':
            denoms_lts[bn_id] = "noun."+str(count_noun)
            count_noun = count_noun + 1
        elif bn_id[-1]=='v':
            denoms_lts[bn_id] = 'verb.' + str(count_verb)
            count_verb = count_verb + 1
        elif bn_id[-1]=='a':
            denoms_lts[bn_id] = 'adj.' + str(count_adj)
            count_adj = count_adj + 1
        elif bn_id[-1]=='r':
            denoms_lts[bn_id] = 'adv.' + str(count_adv)
            count_adv = count_adv + 1
    return denoms_lts


def eliminate_secondary_senses(word_element_df, bnids_denoms_dict):
    word_element_df_named = word_element_df.replace(to_replace={'bn_id' : bnids_denoms_dict}, value=None)
    row_check = [(row.bn_id[0] != 'b') for row in word_element_df_named.itertuples()]
    word_element_df_named = word_element_df_named[row_check]
    word_element_df_named = word_element_df_named.rename(columns={"word": "word", "bn_id": "sense",
                                  word_element_df_named.columns[2]:word_element_df_named.columns[2]})
    return word_element_df_named


def assign_senses_to_word(word, input_dbs, output_dbs):
    Utils.init_logging(os.path.join("CreateEntities" ,"SenseDenominations.log"), logging.INFO)

    hdf5_min_itemsizes = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'sense': Utils.HDF5_BASE_SIZE_512 / 16,
                          Utils.DEFINITIONS: Utils.HDF5_BASE_SIZE_512 / 2, Utils.EXAMPLES: Utils.HDF5_BASE_SIZE_512 / 2,
                          Utils.SYNONYMS: Utils.HDF5_BASE_SIZE_512 / 4, Utils.ANTONYMS: Utils.HDF5_BASE_SIZE_512 / 4}

    word_dfs = [input_dbs[i].select(key=Utils.CATEGORIES[i], where="word == " + str(word))
                for i in range(len(Utils.CATEGORIES))]
    bn_ids = set(word_dfs[0]['bn_id'])

    senses_resources_lts = []

    for bn_id in bn_ids:
        senses_df = [word_dfs[i].loc[word_dfs[i]['bn_id']== str(bn_id)]
                    for i in range(len(Utils.CATEGORIES))]
        # sense_examples_df = word_examples_df.loc[word_examples_df['bn_id']== str(bn_id)]
        # sense_synonyms_df = word_synonyms_df.loc[word_synonyms_df['bn_id']== str(bn_id)]
        # sense_antonyms_df = word_antonyms_df.loc[word_antonyms_df['bn_id']== str(bn_id)]

        senses_resources_lts.append((bn_id,
                                     compute_importance_score(
                                        (len(senses_df[0]), len(senses_df[1]), # D, E, S, A
                                         len(senses_df[2]), len(senses_df[3])) )))

    sorted_senses_lts = sorted(senses_resources_lts, key=lambda tpl: tpl[1], reverse=True)
    logging.info(sorted_senses_lts)
    logging.info(len(sorted_senses_lts))

    if len(sorted_senses_lts) >= 15:
        logging.info("Applying high score cutoff")
        sorted_senses_lts = list(filter(lambda tpl: tpl[1] >= HIGH_SCORE_CUTOFF, sorted_senses_lts))
    elif len(sorted_senses_lts) >= 5:
        sorted_senses_lts = list(filter(lambda tpl: tpl[1] >= LOW_SCORE_CUTOFF, sorted_senses_lts))
        logging.info("Applying low score cutoff")

    bnids_denoms_dict = get_denominations(sorted_senses_lts)
    logging.info(bnids_denoms_dict)
    logging.info(len(bnids_denoms_dict.keys()))

    word_dfs_named = [eliminate_secondary_senses(word_dfs[i], bnids_denoms_dict)
                    for i in range(len(Utils.CATEGORIES))]
    # word_examples_df_named = eliminate_secondary_senses(word_examples_df, bnids_denoms_dict)
    # word_synonyms_df_named = eliminate_secondary_senses(word_synonyms_df, bnids_denoms_dict)
    # word_antonyms_df_named = eliminate_secondary_senses(word_antonyms_df, bnids_denoms_dict)

    for i in range(len(Utils.CATEGORIES)):
        
        output_dbs[i].append(key=Utils.CATEGORIES[i], value=word_dfs_named[i],
                         min_itemsize={key:hdf5_min_itemsizes[key]
                                       for key in hdf5_min_itemsizes.keys() if key in ['word', 'sense', Utils.CATEGORIES[i]]})
    # output_dbs[1].append(key=Utils.EXAMPLES, value=word_examples_df_named,
    #                      min_itemsize={key for key in hdf5_min_itemsizes if
    #                                    key in ['word', 'sense', Utils.DEFINITIONS]})
    # output_dbs[2].append(key=Utils.SYNONYMS, value=word_synonyms_df_named,
    #                      min_itemsize={key for key in hdf5_min_itemsizes if
    #                                    key in ['word', 'sense', Utils.DEFINITIONS]})
    # output_dbs[3].append(key=Utils.ANTONYMS, value=word_antonyms_df_named,
    #                      min_itemsize={key for key in hdf5_min_itemsizes if
    #                                    key in ['word', 'sense', Utils.DEFINITIONS]})


def main():
    hdf5_input_filepaths = [os.path.join(Utils.FOLDER_INPUT, Utils.PROCESSED + "_" + categ + ".h5") for categ in Utils.CATEGORIES]
    hdf5_output_filepaths = [os.path.join(Utils.FOLDER_INPUT, Utils.DENOMINATED + '_' + categ + ".h5")
                             for categ in Utils.CATEGORIES]
    input_dbs = [pd.HDFStore(input_fpath, mode='r') for input_fpath in hdf5_input_filepaths]  # D,E,S,A
    output_dbs = [pd.HDFStore(output_fpath, mode='w') for output_fpath in hdf5_output_filepaths]

    mini_vocab = ['wide', 'move', 'plant', 'light']
    for word in mini_vocab:
        logging.info("Selecting, sorting and naming the senses of the word: " + word)
        assign_senses_to_word(word, input_dbs, output_dbs)

    Utils.close_list_of_files(input_dbs + output_dbs)