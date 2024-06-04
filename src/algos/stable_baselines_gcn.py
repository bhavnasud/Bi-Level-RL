import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv


class ActorGCN(nn.Module):
    def __init__(self, hidden_features_dim: int = 8, node_features_dim=2, action_dim=1):
        super(ActorGCN, self).__init__()
        self.conv1 = GCNConv(node_features_dim, node_features_dim)
        self.lin1 = nn.Linear(node_features_dim, hidden_features_dim)
        self.lin2 = nn.Linear(hidden_features_dim, hidden_features_dim)
        self.lin3 = nn.Linear(hidden_features_dim, action_dim)
        self.action_dim = action_dim

    def forward(self, observations) -> torch.Tensor:
        # hack to get a device, TODO fix this
        # device = self.lin1.weight.device
        x = observations['node_features'].type(torch.FloatTensor)
        num_nodes = x.shape[1]
        edge_index = observations['edge_index'].type(torch.LongTensor)
        data_list = [Data(x=x[i], edge_index=edge_index[i]) for i in range(x.shape[0])]
        batch = Batch.from_data_list(data_list)
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = x + batch.x
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        return x.reshape(-1, num_nodes, self.action_dim)

class CriticGCN(nn.Module):
    def __init__(self, hidden_features_dim: int = 8, node_features_dim=2, action_dim=1):
        super(CriticGCN, self).__init__()
        self.conv1 = GCNConv(node_features_dim, node_features_dim)
        self.lin1 = nn.Linear(node_features_dim, hidden_features_dim)
        self.lin2 = nn.Linear(hidden_features_dim, hidden_features_dim)
        # self.lin3 = nn.Linear(hidden_features_dim, 1)

    def forward(self, observations, actions=None) -> torch.Tensor:
        # hack to get a device, TODO fix this
        # device = self.lin1.weight.device
        x = observations['node_features'].type(torch.FloatTensor)
        num_nodes = x.shape[1]
        # if actions is not None:
        #     x = torch.cat([x, actions.unsqueeze(-1)], dim=-1)
        edge_index = observations['edge_index'].type(torch.LongTensor)
        data_list = [Data(x=x[i], edge_index=edge_index[i]) for i in range(x.shape[0])]
        batch = Batch.from_data_list(data_list)
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = x + batch.x
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = x.reshape(-1, num_nodes, x.shape[1])
        # sum features over all nodes
        x = torch.sum(x, dim=1)
        return x
        # return self.lin3(x)
