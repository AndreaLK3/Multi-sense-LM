import torch
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from WordEmbeddings.ComputeEmbeddings import Method

# classRGCNConv(in_channels, out_channels, num_relations, num_bases, root_weight=True, bias=True, **kwargs)
# is the relational graph convolutional operator from “Modeling Relational Data with Graph Convolutional Networks”

# Edge type needs to be a one-dimensional torch.long tensor
# which stores a relation identifier ∈{0,…,||−1} for each edge.

# Parameters:
#   in_channels (int) – Size of each input sample.
#   out_channels (int) – Size of each output sample.
#   num_relations (int) – Number of relations.
#   num_bases (int) – Number of bases used for basis-decomposition.
#                     Due to operating with only 5 relations, I should deactivate this
#   root_weight (bool, optional) – If set to False, the layer will not add transformed root node features to the output.
#                                  (default: True)
#   bias (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)
#
#   **kwargs (optional) – Additional arguments of torch_geometric.nn.conv.MessagePassing.

data = 0 # PLACEHOLDER: must load all the data (single prototype embeddings + dictionary input) as a graph

NUM_RELATIONS = 5

# Modified from: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/rgcn.py
class Net_RGCN(torch.nn.Module):
    def __init__(self, data, method):
        super(Net_RGCN, self).__init__()
        if method == Method.DISTILBERT:
            self.node_state_units = 768
        else: # if method == Method.FASTTEXT:
            self.node_state_units = 300

        self.conv1 = RGCNConv(
            data.num_nodes, self.node_state_units, NUM_RELATIONS, num_bases=NUM_RELATIONS)
        # self.conv2 = RGCNConv(
        #     data.num_nodes, self.node_state_units, NUM_RELATIONS, num_bases=NUM_RELATIONS)

    def forward(self, edge_index, edge_type, edge_norm):
        x = F.relu(self.conv1(None, edge_index, edge_type))
        # x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net_RGCN(data, Method.FASTTEXT).to(device), Net_RGCN.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.edge_index, data.edge_type, data.edge_norm)
    F.nll_loss(out[data.train_idx], data.train_y).backward()
    optimizer.step()


def test():
    model.eval()
    out = model(data.edge_index, data.edge_type, data.edge_norm)
    pred = out[data.test_idx].max(1)[1]
    acc = pred.eq(data.test_y).sum().item() / data.test_y.size(0)
    return acc


for epoch in range(1, 51):
    train()
    test_acc = test()
    print('Epoch: {:02d}, Accuracy: {:.4f}'.format(epoch, test_acc))