import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv



class GCNExtractor(nn.Module):
    def __init__(self, hidden_features_dim: int = 8, node_features_dim=2, action_dim=0):
        super(GCNExtractor, self).__init__()
        self.conv1 = GCNConv(node_features_dim + action_dim, node_features_dim + action_dim)
        self.lin1 = nn.Linear(node_features_dim + action_dim, hidden_features_dim)
        self.lin2 = nn.Linear(hidden_features_dim, hidden_features_dim)
        self.lin3 = nn.Linear(hidden_features_dim, 1)

    def forward(self, observations, actions=None) -> torch.Tensor:
        x = observations['node_features'].type(torch.FloatTensor)
        if actions is not None:
            x = torch.cat([x, actions], dim=-1)
        num_nodes = x.shape[1]
        edge_index = observations['edge_index'].type(torch.LongTensor)
        data_list = [Data(x=x[i], edge_index=edge_index[i]) for i in range(x.shape[0])]
        batch = Batch.from_data_list(data_list)
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = x + batch.x
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = self.lin3(x)
        return_val = x.reshape(-1, num_nodes, 1)
        return return_val
