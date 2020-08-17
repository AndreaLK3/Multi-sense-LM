import torch
from torch_geometric.nn import GATConv
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
import torch.nn.functional as tfunc
from GNN.Models.Common import unpack_input_tensor, init_model_parameters, lemmatize_node
from torch.nn.parameter import Parameter
import logging
import nltk
from PrepareKBInput.LemmatizeNyms import lemmatize_term


class RNN(torch.nn.Module):

    def __init__(self, model_type, data, grapharea_size, grapharea_matrix, vocabulary_wordlist,
                 include_globalnode_input, include_sensenode_input, predict_senses,
                 batch_size, n_layers, n_hid_units, dropout_p):
        super(RNN, self).__init__()
        self.model_type = model_type  # can be "LSTM" or "GRU"
        # init_model_parameters(model, graph_dataobj, grapharea_size, grapharea_matrix, vocabulary_df,
        #                           include_globalnode_input, include_sensenode_input, predict_senses,
        #                           batch_size, n_layers, n_hid_units, dropout_p)
        self.num_embs = data.x.shape[0]
        self.dim_embs = data.x.shape[1]

        range_random_embs = 10
        self.embs_A = (torch.rand((self.num_embs, self.dim_embs)) - 0.5) * range_random_embs
        self.embs_B = (torch.rand((self.num_embs, self.dim_embs)) - 0.5) * range_random_embs
        self.select_first_indices = Parameter(torch.tensor(list(range(n_hid_units))).to(torch.float32),
                                              requires_grad=False)
        self.embedding_zeros = Parameter(torch.zeros(size=(1, self.dim_embs)), requires_grad=False)

        self.network_1 = torch.linear()