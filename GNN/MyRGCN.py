import torch
from torch_geometric.nn import RGCNConv, GCNConv, GatedGraphConv
import Utils
import Filesystem as F
import logging
import torch.nn.functional as tfunc
from time import time
from Utils import DEVICE
from torch.nn.parameter import Parameter

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

class GRU_RGCN(torch.nn.Module):
    def __init__(self, data, grapharea_size, gate_dim):
        super(GRU_RGCN, self).__init__()
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

        # It is necessary to have 2 matrices, update_gate_W ( 32*300 x 300)  and update_gate_U ( 300 x 300)
        self.update_gate_W = Parameter(torch.empty(size=(self.N * self.d, gate_dim)).to(DEVICE), requires_grad=True)
        self.update_gate_U = Parameter(torch.empty(size=(self.d, gate_dim)).to(DEVICE), requires_grad=True)

        self.memory_previous_rgcnconv = torch.zeros(size=(grapharea_size,self.d)).to(DEVICE)

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

            # Update gate: u_v^t = σ(W^u * a_v^t +  U^u * h_v^(t-1)).
            # I don't have h_v^(t-1). In practice, I can use either h_v^t or
            neighbourhood_contribution_update_gate = torch.mm(torch.flatten(x).unsqueeze(dim=0), self.update_gate_W)
            prevstate_contribution_update_gate = torch.mm(
                self.memory_previous_rgcnconv.index_select(dim=0, index=torch.tensor([0]).to(DEVICE)),
                self.update_gate_U)
            update_gate = torch.sigmoid(neighbourhood_contribution_update_gate + prevstate_contribution_update_gate)
            # GRU update: h^{t+1}=u∙(̃h^{t+1}) + (1-u)∙h^t
            rgcn_conv = update_gate * proposed_rgcn_conv + \
                        (torch.tensor(1)-update_gate) * self.memory_previous_rgcnconv
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



##### Some of the alternatives specified & written previously for GNNs and recurrence #####
##### They may be taken out of this "comment freezer" to be used and compared         #####

# class GRU_RGCN_WConv(torch.nn.Module):
#     def __init__(self, data, grapharea_size, gate_dim):
#         super(GRU_RGCN_WConv, self).__init__()
#         self.last_idx_senses = data.node_types.tolist().index(1)
#         self.last_idx_globals = data.node_types.tolist().index(2)
#         self.N = grapharea_size
#         self.d = data.x.shape[1]
#
#         # Representation built using the RGCN mechanism, by combining |R| GCNs and the previousLayer-selfConnection
#         self.convs_ls = torch.nn.ModuleList([GCNConv(in_channels=self.d,
#                               out_channels=gate_dim, bias=False).to(DEVICE) for r in range(data.num_relations)])
#         self.W_0 = Parameter(torch.empty(size=(self.d, self.d)).to(DEVICE), requires_grad=True)
#
#         # GRU: I decide to have a update_gate
#         # The update_gate will be based on (x, edge_index, edge_type), i.e. the input of each batch element
#
#         # Following (partially) the formula: u_v^t = σ(W^u * a_v^t +  U^u * h_v^(t-1) ),
#         #   where a_v^t is just the concatenation of the neighbourhood, a_v^t= A_(v:)^T [h_1^(t−1),…,h_(|V|)^(t−1) ] + b
#         # So for us a_v^t will be the selected graph_area, in order to operate on fixed input dimensions.
#
#         # It is necessary to have 2 matrices, update_gate_W (GCN-Conv)  and update_gate_U (300 x 1)
#         self.update_gate_W = GCNConv(in_channels=self.d,
#                               out_channels=self.d, bias=False).to(DEVICE)
#         self.update_gate_U = Parameter(torch.empty(size=(self.d, gate_dim)).to(DEVICE), requires_grad=True)
#
#         self.memory_previous_rgcnconv = torch.zeros(size=(grapharea_size,self.d)).to(DEVICE)
#
#         # 2nd part of the network as before: 2 linear layers from the RGCN representation to the logits
#         self.linear2global = torch.nn.Linear(in_features=self.d,
#                                              out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
#         self.linear2sense = torch.nn.Linear(in_features=self.d, out_features=self.last_idx_senses, bias=True)
#
#         # Once the structure has been specified, we initialize the Parameters we defined
#         [torch.nn.init.xavier_normal_(my_param) for my_param in [self.W_0, self.update_gate_U]]
#
#
#     def forward(self, batchinput_ls):  # given the batches, the current node is at index 0
#         predictions_globals_ls = []
#         predictions_senses_ls = []
#         # T-BPTT: at the start of each batch, we detach_() the hidden state from the graph&history that created it
#         self.memory_previous_rgcnconv.detach_()
#
#         for (x, edge_index, edge_type) in batchinput_ls:
#
#             (split_sources, split_destinations) = split_edge_index(edge_index, edge_type)
#             split_edge_index_ls = []
#             for i in range(len(split_sources)):
#                 split_edge_index_ls.append(torch.stack([split_sources[i], split_destinations[i]]))
#
#             rels_gcnconv_output_ls = [self.convs_ls[i](x, split_edge_index_ls[i]) for i in range(len(split_edge_index_ls))]
#
#             A0_selfadj = torch.eye(self.N).to(DEVICE)
#             prevlayer_connection = torch.mm(A0_selfadj, torch.mm(x, self.W_0))
#             composite_rgcn_conv = sum(rels_gcnconv_output_ls)
#             # adding contribution from h_v^(l-1), the previous layer of the same node
#             proposed_rgcn_conv = composite_rgcn_conv + prevlayer_connection
#
#             # Update gate: u_v^t = σ(W^u * a_v^t +  U^u * h_v^(t-1)).
#             # I don't have h_v^(t-1). In practice, I can use either h_v^t or
#             neighbourhood_contribution_update_gate = self.update_gate_W(x, edge_index)
#             prevstate_contribution_update_gate = torch.mm(
#                 self.memory_previous_rgcnconv.index_select(dim=0, index=torch.tensor([0]).to(DEVICE)),
#                 self.update_gate_U)
#             update_gate = torch.sigmoid(neighbourhood_contribution_update_gate + prevstate_contribution_update_gate)
#             # GRU update: h^{t+1}=u∙(̃h^{t+1}) + (1-u)∙h^t
#             rgcn_conv = update_gate * proposed_rgcn_conv + \
#                         (torch.tensor(1)-update_gate) * self.memory_previous_rgcnconv
#             self.memory_previous_rgcnconv = rgcn_conv.clone() # store h in memory
#
#             x_Lplus1 = tfunc.relu(rgcn_conv)
#
#             x1_current_node = x_Lplus1[0]  # current_node_index
#             logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
#             logits_sense = self.linear2sense(x1_current_node)
#
#             sample_predictions_globals = tfunc.log_softmax(logits_global, dim=0)
#             predictions_globals_ls.append(sample_predictions_globals)
#             sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
#             predictions_senses_ls.append(sample_predictions_senses)
#
#         return torch.stack(predictions_globals_ls, dim=0), torch.stack(predictions_senses_ls, dim=0)


# # Executing separately the convolution on for each relation, using pre-made Gated GCNs
# class CompositeGGCN(torch.nn.Module):
#     def __init__(self, data, grapharea_size):
#         super(CompositeGGCN, self).__init__()
#         self.last_idx_senses = data.node_types.tolist().index(1)
#         self.last_idx_globals = data.node_types.tolist().index(2)
#         self.N = grapharea_size
#         self.d = data.x.shape[1]
#
#         self.gated_convs_ls = torch.nn.ModuleList([GatedGraphConv(out_channels=data.x.shape[1], num_layers=1,
#                                                                   bias=False).to(DEVICE) for r in range(data.num_relations)])
#         self.W_0 = Parameter(torch.empty(size=(self.d, self.d)), requires_grad=True)
#         torch.nn.init.xavier_normal_(self.W_0)
#
#         # 2nd part of the network as before: 2 linear layers from the RGCN representation to the logits
#         self.linear2global = torch.nn.Linear(in_features=self.d,
#                                              out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
#         self.linear2sense = torch.nn.Linear(in_features=self.d, out_features=self.last_idx_senses, bias=True)
#
#     def forward(self, batchinput_ls):  # given the batches, the current node is at index 0
#         predictions_globals_ls = []
#         predictions_senses_ls = []
#         for (x, edge_index, edge_type) in batchinput_ls:
#
#             (split_sources, split_destinations) = split_edge_index(edge_index, edge_type)
#             split_edge_index_ls = []
#             for i in range(len(split_sources)):
#                 split_edge_index_ls.append(torch.stack([split_sources[i], split_destinations[i]]))
#
#             rels_gcnconv_output_ls = [self.gated_convs_ls[i](x, split_edge_index_ls[i]) for i in range(len(split_edge_index_ls))]
#
#             A0_selfadj = torch.eye(self.N).to(DEVICE)
#             prevlayer_connection = torch.mm(A0_selfadj, torch.mm(x, self.W_0))
#             relgcn_conv = sum(rels_gcnconv_output_ls)
#             # adding contribution from h_v^(l-1), the previous layer of the same node
#             relgcn_conv = relgcn_conv + prevlayer_connection
#
#             x_Lplus1 = tfunc.leaky_relu(relgcn_conv, negative_slope=0.1)
#
#             x1_current_node = x_Lplus1[0]  # current_node_index
#             logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
#             logits_sense = self.linear2sense(x1_current_node)
#
#             sample_predictions_globals = tfunc.log_softmax(logits_global, dim=0)
#             predictions_globals_ls.append(sample_predictions_globals)
#             sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
#             predictions_senses_ls.append(sample_predictions_senses)
#
#         return torch.stack(predictions_globals_ls, dim=0), torch.stack(predictions_senses_ls, dim=0)

# # Executing separately the convolution on for each relation, I use the pre-made standard GCNs
# class RGCN(torch.nn.Module):
#     def __init__(self, data, grapharea_size):
#         super(RGCN, self).__init__()
#         self.last_idx_senses = data.node_types.tolist().index(1)
#         self.last_idx_globals = data.node_types.tolist().index(2)
#         self.N = grapharea_size
#         self.d = data.x.shape[1]
#
#         self.convs_ls = torch.nn.ModuleList([GCNConv(in_channels=data.x.shape[1],
#                               out_channels=data.x.shape[1], bias=False).to(DEVICE) for r in range(data.num_relations)])
#         self.W_0 = Parameter(torch.empty(size=(self.d, self.d)), requires_grad=True)
#         torch.nn.init.xavier_normal_(self.W_0)
#
#         # 2nd part of the network as before: 2 linear layers from the RGCN representation to the logits
#         self.linear2global = torch.nn.Linear(in_features=self.d,
#                                              out_features=self.last_idx_globals - self.last_idx_senses, bias=True)
#         self.linear2sense = torch.nn.Linear(in_features=self.d, out_features=self.last_idx_senses, bias=True)
#
#     def forward(self, batchinput_ls):  # given the batches, the current node is at index 0
#         predictions_globals_ls = []
#         predictions_senses_ls = []
#         for (x, edge_index, edge_type) in batchinput_ls:
#
#             (split_sources, split_destinations) = split_edge_index(edge_index, edge_type)
#             split_edge_index_ls = []
#             for i in range(len(split_sources)):
#                 split_edge_index_ls.append(torch.stack([split_sources[i], split_destinations[i]]))
#
#             rels_gcnconv_output_ls = [self.convs_ls[i](x, split_edge_index_ls[i]) for i in range(len(split_edge_index_ls))]
#
#             A0_selfadj = torch.eye(self.N).to(DEVICE)
#             prevlayer_connection = torch.mm(A0_selfadj, torch.mm(x, self.W_0))
#             relgcn_conv = sum(rels_gcnconv_output_ls)
#             # adding contribution from h_v^(l-1), the previous layer of the same node
#             relgcn_conv = relgcn_conv + prevlayer_connection
#
#             x_Lplus1 = tfunc.leaky_relu(relgcn_conv, negative_slope=0.1)
#
#             x1_current_node = x_Lplus1[0]  # current_node_index
#             logits_global = self.linear2global(x1_current_node)  # shape=torch.Size([5])
#             logits_sense = self.linear2sense(x1_current_node)
#
#             sample_predictions_globals = tfunc.log_softmax(logits_global, dim=0)
#             predictions_globals_ls.append(sample_predictions_globals)
#             sample_predictions_senses = tfunc.log_softmax(logits_sense, dim=0)
#             predictions_senses_ls.append(sample_predictions_senses)
#
#         return torch.stack(predictions_globals_ls, dim=0), torch.stack(predictions_senses_ls, dim=0)