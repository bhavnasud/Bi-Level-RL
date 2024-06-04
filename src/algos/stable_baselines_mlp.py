from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, LeakyReLU
from torch_geometric.data import Data, Batch

class ActorMlp(nn.Module):
    def __init__(self, hidden_features_dim: int = 8, node_features_dim=2, edge_features_dim=1,
                 action_dim=1, num_nodes=1):
        super(ActorMlp, self).__init__()
        self.num_nodes = num_nodes
        self.lin1 = nn.Linear((num_nodes * node_features_dim) + edge_features_dim, hidden_features_dim)
        self.lin2 = nn.Linear(hidden_features_dim, hidden_features_dim)
        self.lin3 = nn.Linear(hidden_features_dim, hidden_features_dim)
        self.lin4 = nn.Linear(hidden_features_dim, action_dim)
        self.action_dim = action_dim

    def forward(self, observations) -> torch.Tensor:
        x = observations['node_features'].type(torch.IntTensor)
        edge_index = observations['edge_index'].type(torch.LongTensor)
        edge_attr = observations['edge_features']
        data_list = [Data(x=x[i], edge_index=edge_index[i], edge_attr=edge_attr[i]) for i in range(x.shape[0])]
        batch = Batch.from_data_list(data_list)
        input = torch.cat(batch.x + batch.edge_attr)
        x = F.leaky_relu(self.lin1(input))
        x = F.leaky_relu(self.lin2(x))
        x = F.leaky_relu(self.lin3(x))
        x = F.softplus(self.lin4(x))
        return x.reshape(-1, self.num_nodes, self.action_dim)
    
class CriticMlp(nn.Module):
    def __init__(self, hidden_features_dim: int = 8, node_features_dim=2, edge_features_dim=1, action_dim=0,
                 num_nodes=1):
        super(CriticMlp, self).__init__()
        self.num_nodes = num_nodes
        self.lin1 = nn.Linear((num_nodes) * (node_features_dim + action_dim) + edge_features_dim, hidden_features_dim)
        self.lin2 = nn.Linear(hidden_features_dim, hidden_features_dim)
        self.lin3 = nn.Linear(hidden_features_dim, hidden_features_dim)
        self.lin4 = nn.Linear(hidden_features_dim, 1)

    def forward(self, observations, actions=None) -> torch.Tensor:
        x = observations['node_features'].type(torch.IntTensor)
        if actions is not None:
            x = torch.cat([x, actions.unsqueeze(-1)], dim=-1)
        edge_attr = observations['edge_features']
        input = torch.cat([x, edge_attr])
        x = F.leaky_relu(self.lin1(input))
        x = F.leaky_relu(self.lin2(x))
        x = F.leaky_relu(self.lin3(x))
        x = self.lin4(x)
        return x.reshape(-1, 1)
