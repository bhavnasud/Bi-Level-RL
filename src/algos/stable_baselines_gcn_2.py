import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomMultiInputExtractorActor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted per node.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space, hidden_features_dim: int = 256, node_features_dim=13, edge_features_dim=1, num_nodes=1):
        super(CustomMultiInputExtractorActor, self).__init__(observation_space, (num_nodes * hidden_features_dim))
        self.conv1 = GCNConv(node_features_dim, node_features_dim)
        self.lin1 = nn.Linear(node_features_dim, hidden_features_dim)
        self.lin2 = nn.Linear(hidden_features_dim, hidden_features_dim)
        # self.lin3 = nn.Linear(hidden_features_dim, 1)
    
    def forward(self, observations) -> torch.Tensor:
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
        x = x.reshape(-1, num_nodes * x.shape[1])
        return x
        # return self.lin3(x)
