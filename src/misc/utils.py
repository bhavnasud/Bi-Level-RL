import numpy as np
from enum import Enum
from src.algos.stable_baselines_mlp import ActorMlp, CriticMlp
# from src.algos.stable_baselines_mpnn import ActorMpnn, CriticMpnn
# from src.algos.stable_baselines_gcn import ActorGCN, CriticGCN

def mat2str(mat):
    return str(mat).replace("'",'"').replace('(','<').replace(')','>').replace('[','{').replace(']','}')  

def dictsum(dic,t):
    return sum([dic[key][t] for key in dic if t in dic[key]])

def moving_average(a, n=3) :
    """
    Computes a moving average used for reward trace smoothing.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class FeatureExtractor(Enum):
    MLP = 0
    GCN = 1
    MPNN = 2


class RLAlgorithm(Enum):
    A2C = 0
    PPO = 1
    SAC = 2

   
def get_rl_network(extractor_type, hidden_features_dim, node_features_dim, edge_features_dim,
                          num_nodes, action_dim, value_net=False, value_net_include_actions=False):
    if extractor_type == FeatureExtractor.MLP:
        if value_net:
            return CriticMlp(hidden_features_dim=hidden_features_dim, node_features_dim=node_features_dim,
                            edge_features_dim=edge_features_dim, action_dim=action_dim if value_net_include_actions else 0,
                            num_nodes=num_nodes)
        else:
            return ActorMlp(hidden_features_dim=hidden_features_dim, node_features_dim=node_features_dim,
                            edge_features_dim=edge_features_dim, action_dim=action_dim,
                            num_nodes=num_nodes)
    elif extractor_type == FeatureExtractor.MPNN:
        if value_net:
            return CriticMpnn(hidden_features_dim=hidden_features_dim, node_features_dim=node_features_dim,
                            edge_features_dim=edge_features_dim, action_dim=action_dim if value_net_include_actions else 0)
        else:
            return ActorMpnn(hidden_features_dim=hidden_features_dim, node_features_dim=node_features_dim,
                            edge_features_dim=edge_features_dim, action_dim=action_dim)
    else:
        if value_net:
            return CriticGCN(hidden_features_dim=hidden_features_dim, node_features_dim=node_features_dim,
                            action_dim=action_dim if value_net_include_actions else 0)
        else:
            return ActorGCN(hidden_features_dim=hidden_features_dim, node_features_dim=node_features_dim,
                            action_dim=action_dim)
