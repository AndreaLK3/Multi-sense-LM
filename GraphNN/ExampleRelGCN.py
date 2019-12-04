import torch
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from WordEmbeddings.ComputeEmbeddings import Method
import random
import numpy as np

# In this example, we do not extract the node features, word and sense vocabulary indices, etc.
# We use Random Number Generation to create a small structure.

# 1 definition (or example) refers to only 1 sense. 1 sense can have multiple definitions or examples
def setrandomedges_d_or_e(senses_low, senses_high, elements_low, elements_high):
    num_elements = elements_high - elements_low
    possible_elements_ls = list(range(elements_low, elements_high))
    sense_deflinks_dict = {}
    for sense_index in range(senses_low, senses_high):
        num_defs_for_sense = random.randint(1,1+num_elements//4)
        sense_defs_ls = np.random.choice(possible_elements_ls, num_defs_for_sense, replace=False)
        possible_elements_ls = list( set(possible_elements_ls).difference(set(sense_defs_ls)))
        sense_deflinks_dict[sense_index] = sense_defs_ls
    return sense_deflinks_dict

# 1 sense refers to only 1 global. 1 global can have multiple senses
def setrandomedges_sc(senses_low, senses_high, global_low, global_high):
    # same case, just with senses in the other place:
    return setrandomedges_d_or_e(global_low, global_high, senses_low, senses_high)


# Any global can have multiple synonyms (or antonyms). However, they must be symmetrical (a,b) ==> (b,a).
def setrandomedges_syn_or_ant(global_low, global_high):
    for global_index in range(global_low, global_high):

    pass



def createInputGraph():

    num_senses = 10
    num_sp = 3
    num_def = 15
    num_exs = 10
    NUM_NODES = num_senses + num_sp + num_def + num_exs
    NUM_NODE_FEATURES = 30
    # X (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
    #
    # The nodes are: all the sense embeddings + all the single-prototype embeddings
    #               + all the sentence embeddings from the definitions and examples.
    # All the nodes must have an index, from 0 to num_nodes -1, and the same dimensionality, num_node_features.
    X_senses = torch.rand((num_senses, NUM_NODE_FEATURES))
    X_sp = torch.rand((num_sp, NUM_NODE_FEATURES))
    X_defs = torch.rand((num_def, NUM_NODE_FEATURES))
    X_exs = torch.rand((num_exs, NUM_NODE_FEATURES))

    # The order for the index of the nodes?
    # sense = [0,se) ; single prototype = [se,se+sp) ; definitions = [se+sp, se+sp+d) ; examples = [se+sp+d, e==num_nodes)
    # The index will be the row number in the matrix X

    # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
    # We can operate with a list of S-D tuples, adding t().contiguous()
    # The Procedure to set up both X and edge_index will be:
    # - read the archive of processed definitions. We encounter the sense_wn_id (e.g. active.n.03), and the def. text
    # - consider the database indices_table.sql. It has the columns: word_sense, vocab_index (from 0 to se),
    #   start_defs, end_defs, start_examples, end_examples.
    # - load (for instance) vectorized_FastText_definitions.npy. Even with > 20K vectors, it is very light at ~58MB
    # - (Using WordNet as a source, we always have 1 definition). Use [start_defs, end_defs) to extract the vector(s) for
    #   the definition of a sense, and append that vector to X_defs
    # - Register the connections: add a tuple (source=definition_index=sp+def, target=sense)
    # - Do the same for examples: get the examples' vectors for that sense from [start_examples, end_examples) in
    #   indices_table.sql, and get the corresponding rows of vectorized_FastText_examples.npy.
    # - Append vectors to submatrix X_exs (to be put together later), and add the connections [d, e==num_nodes) ->[0,se)

    # For this example, however, we initialized randomly the features of the vectors, and we proceed by
    # randomly determining connections between different kinds of nodes, in the way the task allows.
    # definitions -> senses : [se+sp, se+sp+d) -> [0,se)
    edges_defs_senses = setrandomedges_d_or_e(0, num_senses, num_senses+num_sp, num_senses+num_sp+num_def)

    # examples --> senses : [se+sp+d, e==num_nodes) -> [0,se)
    edges_exs_senses = setrandomedges_d_or_e(0, num_senses, num_senses+num_sp+num_def, NUM_NODES)

    # global (a.k.a. single-prototype) -> senses : [se,se+sp) -> [0,se)
    edges_sp_senses = [torch.randint(num_sense, num_sense + num_sp, (num_sp,)),
                       torch.randint(0, num_sense, (num_sense,))]

    # global -> global : [se,se+sp) -> [se,se+sp). Bidirectional (which means 2 connections, (a,b) and (b,a)
    edges_sp_sp_synonyms = [torch.randint(num_sense, num_sense + num_sp, (num_sp,)),
                            torch.randint(num_sense, num_sense + num_sp, (num_sp,))]
    edges_sp_sp_antonyms = [torch.randint(num_sense, num_sense + num_sp, (num_sp,)),
                            torch.randint(num_sense, num_sense + num_sp, (num_sp,))]