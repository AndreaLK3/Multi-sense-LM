import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

#  We show a simple example of an unweighted and undirected graph with three nodes and four edges.
#  Each node contains exactly one feature:
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data_01 = Data(x=x, edge_index=edge_index)


# Note that edge_index, the tensor defining the source and target nodes of all edges, is NOT a list of index tuples.
# If you want to write your indices this way, you should transpose and call contiguous on it
# before passing them to the data constructor:
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data_01b = Data(x=x, edge_index=edge_index.t().contiguous())

# Although the graph has only two edges, we need to define four index tuples to account for both directions of a edge.
# We can also access: data.num_nodes, num_edges, num_node_features,
# contains_isolated_nodes(), contains_self_loops(), is_directed()

# Using the Cora dataset in the example
dataset = Planetoid(root='/tmp/Cora', name='Cora')
# A Dataset contains one or more Data graph-objects, with train_mask, val_mask, test_mask

# A 2-layer GCN:
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
# The constructor defines two GCNConv layers which get called in the forward pass of our network.
# Note that the non-linearity  /here, ReLU) is not integrated in the conv calls and hence needs to be applied afterwards
# Finally, the output is a softmax distribution over the number of classes

# Let’s train this model on the train nodes for 200 epochs:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()