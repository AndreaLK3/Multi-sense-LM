import torch
from torch_geometric.nn import RGCNConv, GCNConv
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tfunc
from time import time
from Utils import DEVICE
from torch.nn.parameter import Parameter

### RGCN, using the torch-geometric rgcn-conv implementation. Currently, it has:
###     1 RGCN layer that operates on the selected area of the the graph
###     2 linear layers, that go from the RGCN representation to the global classes and the senses' classes
class PremadeRGCN(torch.nn.Module):
    def __init__(self, data):
        super(PremadeRGCN, self).__init__()
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.conv1 = RGCNConv(in_channels=data.x.shape[1], # doc: "Size of each input sample " in the example, num_nodes
                              out_channels=data.x.shape[1], # doc: "Size of each output sample "
                              num_relations=data.num_relations,
                              num_bases=data.num_relations)
        self.linear2global = torch.nn.Linear(in_features=data.x.shape[1],
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        self.linear2sense = torch.nn.Linear(in_features=data.x.shape[1], out_features=self.last_idx_senses, bias=True)

    def forward(self, batchinput_ls):  # given the batches, the current node is at index 0
        predictions_globals_ls = []
        predictions_senses_ls = []
        for (x, edge_index, edge_type) in batchinput_ls:
            rgcn_conv = self.conv1(x, edge_index, edge_type)
            # normalizer = batchnorm.BatchNorm1d(num_features=x.shape[1])
            # normalized_rgcn_conv = normalizer(rgcn_conv)
            x_Lplus1 = tfunc.relu(rgcn_conv)
            x1_current_node = x_Lplus1[0]  # current_node_index
            logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
            logits_sense = self.linear2sense(x1_current_node)

            sample_predictions_globals = tfunc.log_softmax(logits_global, dim=0)
            predictions_globals_ls.append(sample_predictions_globals)
            sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
            predictions_senses_ls.append(sample_predictions_senses)

        return torch.stack(predictions_globals_ls, dim=0), torch.stack(predictions_senses_ls, dim=0)


######## Functions for manual GCN convolution, and Rel-GCN
# b_L : N x d
def gcn_convolution(H, A, W_L, b_L):
    support = torch.mm(H, W_L)
    gcn_conv_result = torch.mm(A, support) + b_L
    return gcn_conv_result


def rgcn_convolution(H, Ar_ls, W_all, b_all):
    N = H.shape[0]
    d = H.shape[1]

    # Ar_all is the list of adjacency matrices for the different kinds of edges (/subgraphs).
    # we add here the 0-th, that will be used for W_0^l * h_i^l
    A0 = torch.diag(torch.ones(size=(H.shape[0],)))

    Ar_all = torch.stack([A0] + Ar_ls).to(DEVICE) # prepend

    self_connection = gcn_convolution(H, Ar_all[0], W_all[0], b_all[0])
    
    sum = 0
    for r in range(1, Ar_all.shape[0]):
        A_r = Ar_all[r]
        #logging.info("A_r.shape=" + str(A_r.shape))
        W_r = W_all[r]
        b_r = b_all[r]
        rel_contribution = gcn_convolution(H, A_r, W_r, b_r)
        #logging.info("rel_contribution.shape=" + str(rel_contribution.shape))
        c_ir_s = torch.nonzero(A_r).t()[0].bincount(minlength=N).unsqueeze(0).t().repeat((1,d)).to(DEVICE)
        c_ir_s[c_ir_s==0] = 1 # avoid division by 0 if the node has no neighbours (mainly due to the edges' directions)
        #logging.info("c_ir_s.shape=" + str(c_ir_s.shape))
        sum = sum + rel_contribution / c_ir_s

    sum = sum + self_connection

    return sum
######

######## Tools to split the input of the forward call, (x, edge_index, edge_type),
######## into subgraphs using different adjacency matrices.

def split_edge_index(edge_index, edge_type):

    sections_cutoffs = [i for i in range(edge_type.shape[0]-1) if edge_type[i] != edge_type[i-1]] + [edge_type.shape[0]]
    sections_lengths = [sections_cutoffs[i+1] - sections_cutoffs[i] for i in range(len(sections_cutoffs)-1)]

    split_sources = torch.split(edge_index[0], sections_lengths)
    split_destinations = torch.split(edge_index[1], sections_lengths)

    return (split_sources, split_destinations)


def get_adj_matrix(sources, destinations, grapharea_size):
    A = torch.zeros(size=(grapharea_size, grapharea_size))

    for e in range(sources.shape[0]):
        i = sources[e]
        j = destinations[e]

        A[i][j]=1
    return A


def create_adj_matrices(x, edge_index, edge_type):
    grapharea_size = x.shape[0]

    (split_sources, split_destinations) = split_edge_index(edge_index, edge_type)
    A_ls = []
    for seg in range(len(split_sources)):
        sources = split_sources[seg]
        destinations = split_destinations[seg]
        A_ls.append(get_adj_matrix(sources, destinations, grapharea_size))

    return A_ls
######

### RGCN, using my implementation. Splits the forward()input into different adjacency matrices.
### The weights matrices are defined manually, with no basis decomposition. The structure is equivalent to PremadeRGCN:
###     1 RGCN layer that operates on the selected area of the the graph
###     2 linear layers, that go from the RGCN representation to the global classes and the senses' classes
class MyNetRGCN(torch.nn.Module):
    def __init__(self, data, grapharea_size):
        super(MyNetRGCN, self).__init__()
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]

        # Weights matrices Wr, for the GCN convolution of each relation. W0 is for the previous layer's connection
        self.Wr_ls = []
        self.biasr_ls = []
        for r in range(data.num_relations+1):
            self.W_r = torch.empty(size=(self.d,self.d))
            torch.nn.init.xavier_normal_(self.W_r)
            self.Wr_ls.append(self.W_r)
            # bias
            self.bias_r = torch.empty(size=(self.N,self.d))
            torch.nn.init.xavier_normal_(self.bias_r)
            self.biasr_ls.append(self.bias_r)
        self.Wr_all = Parameter(torch.stack(self.Wr_ls).to(DEVICE), requires_grad=True)
        self.biasr_all = Parameter(torch.stack(self.biasr_ls).to(DEVICE), requires_grad=True)

        # 2nd part of the network is as before: 2 linear layers from the RGCN representation to the logits
        self.linear2global = torch.nn.Linear(in_features=self.d,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        self.linear2sense = torch.nn.Linear(in_features=self.d, out_features=self.last_idx_senses, bias=True)

    def forward(self, batchinput_ls):  # given the batches, the current node is at index 0
        predictions_globals_ls = []
        predictions_senses_ls = []
        for (x, edge_index, edge_type) in batchinput_ls:

            Ar_ls = create_adj_matrices(x, edge_index, edge_type)
            rgcn_conv = rgcn_convolution(x, Ar_ls, self.Wr_all, self.biasr_all)
            x_Lplus1 = tfunc.relu(rgcn_conv)

            x1_current_node = x_Lplus1[0]  # current_node_index
            logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
            logits_sense = self.linear2sense(x1_current_node)
            sample_predictions_globals = tfunc.log_softmax(logits_global, dim=0)
            predictions_globals_ls.append(sample_predictions_globals)
            sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
            predictions_senses_ls.append(sample_predictions_senses)

        return torch.stack(predictions_globals_ls, dim=0), torch.stack(predictions_senses_ls, dim=0)



# Executing separately the convolution on for each relation, I use the pre-made standard GCNs
class CompositeRGCN(torch.nn.Module):
    def __init__(self, data, grapharea_size):
        super(CompositeRGCN, self).__init__()
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]

        self.convs_ls = torch.nn.ModuleList([GCNConv(in_channels=data.x.shape[1],
                              out_channels=data.x.shape[1], bias=False).to(DEVICE) for r in range(data.num_relations)])
        self.W_0 = Parameter(torch.empty(size=(self.d, self.d)), requires_grad=True)
        torch.nn.init.xavier_normal_(self.W_0)

        # 2nd part of the network as before: 2 linear layers from the RGCN representation to the logits
        self.linear2global = torch.nn.Linear(in_features=self.d,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        self.linear2sense = torch.nn.Linear(in_features=self.d, out_features=self.last_idx_senses, bias=True)

    def forward(self, batchinput_ls):  # given the batches, the current node is at index 0
        predictions_globals_ls = []
        predictions_senses_ls = []
        for (x, edge_index, edge_type) in batchinput_ls:

            (split_sources, split_destinations) = split_edge_index(edge_index, edge_type)
            split_edge_index_ls = []
            for i in range(len(split_sources)):
                split_edge_index_ls.append(torch.stack([split_sources[i], split_destinations[i]]))

            rels_gcnconv_output_ls = [self.convs_ls[i](x, split_edge_index_ls[i]) for i in range(len(split_edge_index_ls))]

            A0_selfadj = torch.eye(self.N).to(DEVICE)
            prevlayer_connection = torch.mm(A0_selfadj, torch.mm(x, self.W_0))
            relgcn_conv = sum(rels_gcnconv_output_ls)
            # adding contribution from h_v^(l-1), the previous layer of the same node
            relgcn_conv = relgcn_conv + prevlayer_connection

            x_Lplus1 = tfunc.leaky_relu(relgcn_conv, negative_slope=0.1)

            x1_current_node = x_Lplus1[0]  # current_node_index
            logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
            logits_sense = self.linear2sense(x1_current_node)

            sample_predictions_globals = tfunc.log_softmax(logits_global, dim=0)
            predictions_globals_ls.append(sample_predictions_globals)
            sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
            predictions_senses_ls.append(sample_predictions_senses)

        return torch.stack(predictions_globals_ls, dim=0), torch.stack(predictions_senses_ls, dim=0)


class CompositeOneGRU(torch.nn.Module):
    def __init__(self, data, grapharea_size):
        super(CompositeOneGRU, self).__init__()
        self.last_idx_senses = data.node_types.tolist().index(1)
        self.last_idx_globals = data.node_types.tolist().index(2)
        self.N = grapharea_size
        self.d = data.x.shape[1]

        # Representation built using the RGCN mechanism, by combining |R| GCNs and the previousLayer-selfConnection
        self.convs_ls = torch.nn.ModuleList([GCNConv(in_channels=data.x.shape[1],
                              out_channels=data.x.shape[1], bias=False).to(DEVICE) for r in range(data.num_relations)])
        self.W_0 = Parameter(torch.empty(size=(self.d, self.d)).to(DEVICE), requires_grad=True)

        # GRU: I decide to have a update_gate
        # The update_gate will be based on (x, edge_index, edge_type), i.e. the input of each batch element

        # Following (partially) the formula: u_v^t = σ(W^u * a_v^t +  U^u * h_v^(t-1) ),
        #   where a_v^t is just the concatenation of the neighbourhood, a_v^t= A_(v:)^T [h_1^(t−1),…,h_(|V|)^(t−1) ] + b
        # So for us a_v^t will be the selected graph_area, in order to operate on fixed input dimensions.

        # It is necessary to have 2 matrices, update_gate_W ( 32*300 x 1 )  and update_gate_U ( 300 x 1)
        self.update_gate_W = Parameter(torch.empty(size=(self.N * self.d,1)).to(DEVICE), requires_grad=True)
        self.update_gate_U = Parameter(torch.empty(size=(self.d, 1)).to(DEVICE), requires_grad=True)

        self.update_gate = Parameter(torch.ones(size=(1,)).to(DEVICE), requires_grad=True)
        self.memory_previous_rgcnconv = torch.zeros(size=(self.d,)).to(DEVICE)

        # 2nd part of the network as before: 2 linear layers from the RGCN representation to the logits
        self.linear2global = torch.nn.Linear(in_features=self.d,
                                             out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
        self.linear2sense = torch.nn.Linear(in_features=self.d, out_features=self.last_idx_senses, bias=True)

        # Once the structure has been specified, we initialize the Parameters we defined
        [torch.nn.init.xavier_normal_(my_param) for my_param in [self.W_0,
                                                                self.update_gate_W, self.update_gate_U]]


    def forward(self, batchinput_ls):  # given the batches, the current node is at index 0
        predictions_globals_ls = []
        predictions_senses_ls = []
        # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
        self.memory_previous_rgcnconv.detach_()

        for (x, edge_index, edge_type) in batchinput_ls:

            (split_sources, split_destinations) = split_edge_index(edge_index, edge_type)
            split_edge_index_ls = []
            for i in range(len(split_sources)):
                split_edge_index_ls.append(torch.stack([split_sources[i], split_destinations[i]]))

            rels_gcnconv_output_ls = [self.convs_ls[i](x, split_edge_index_ls[i]) for i in range(len(split_edge_index_ls))]

            A0_selfadj = torch.eye(self.N).to(DEVICE)
            prevlayer_connection = torch.mm(A0_selfadj, torch.mm(x, self.W_0))
            composite_rgcn_conv = sum(rels_gcnconv_output_ls)
            # adding contribution from h_v^(l-1), the previous layer of the same node
            proposed_rgcn_conv = composite_rgcn_conv + prevlayer_connection

            # GRU update: h^{t+1}=u∙(̃h^{t+1}) + (1-u)∙h^t
            rgcn_conv = self.update_gate * proposed_rgcn_conv + \
                        (torch.tensor(1)-self.update_gate) * self.memory_previous_rgcnconv
            self.memory_previous_rgcnconv = rgcn_conv.clone() # store h in memory

            x_Lplus1 = tfunc.relu(rgcn_conv)

            x1_current_node = x_Lplus1[0]  # current_node_index
            logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
            logits_sense = self.linear2sense(x1_current_node)

            sample_predictions_globals = tfunc.log_softmax(logits_global, dim=0)
            predictions_globals_ls.append(sample_predictions_globals)
            sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
            predictions_senses_ls.append(sample_predictions_senses)

        return torch.stack(predictions_globals_ls, dim=0), torch.stack(predictions_senses_ls, dim=0)