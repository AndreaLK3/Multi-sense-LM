import torch
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from WordEmbeddings.ComputeEmbeddings import Method


# In this example, we do not extract the node features, word and sense vocabulary indices, etc.
# We use Random Number Generation to create a small structure.

def createInputGraph():

    number_sense_embs = 10
    number_sp_embs = 3
    number_def_sentembs = 15
    number_exs_sentembs = 10
    NUM_NODE_FEATURES = 30
    # X (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
    #
    # The nodes are: all the sense embeddings + all the single-prototype embeddings
    #               + all the sentence embeddings from the definitions and examples.
    # All the nodes must have an index, from 0 to num_nodes -1, and the same dimensionality, num_node_features.
    single_prototype_embeddings = torch.rand((number_sp_embs, NUM_NODE_FEATURES))
    sense_embeddings = torch.rand((number_sense_embs, NUM_NODE_FEATURES))
    definitions_embeddings = torch.rand((number_def_sentembs, NUM_NODE_FEATURES))
    examples_embeddings = torch.rand((number_exs_sentembs, NUM_NODE_FEATURES))

    # The order for the index of the nodes?
    # sense = [0,se) ; single prototype = [se,sp) ; definitions = [sp, d) ; examples = [d, e==num_nodes)
    # The index will be the row number in the matrix X

    # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
    # We can operate with a list of S-D tuples, adding t().contiguous()
    # Procedure:
    # - read the archive of processed definitions. We encounter the sense_wn_id (e.g. active.n.03), and the def. text
    # - consider the database indices_table.sql. It has the columns: 
    #
