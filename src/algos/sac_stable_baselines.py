from gymnasium import spaces
from typing import List, Tuple, Any, Dict, Optional, Union, Type, TypeVar
import torch.nn.functional as F
import torch
import torch.nn as nn

from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.policies import ContinuousCritic, BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from src.algos.dirichlet_distribution import DirichletDistribution
from src.misc.utils import get_rl_network, FeatureExtractor
from src.algos.stable_baselines_gcn_2 import CustomMultiInputExtractorActor

class CustomCriticNetwork(BaseModel):
    def __init__(self, observation_space, action_space, features_dim=128):
        super(CustomCriticNetwork, self).__init__(observation_space, action_space)
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim

        # Define the network architecture for the critics
        # input_dim = observation_space.shape[0] + action_space.shape[0]
        self.network = get_rl_network(extractor_type=FeatureExtractor.GCN, hidden_features_dim=256,
                                node_features_dim=13, edge_features_dim=1,
                                num_nodes=10, action_dim=1, value_net=True)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Concatenate observation and action
        x = torch.cat([obs, action], dim=1)
        features = self.network(x)
        q_value = self.q_value(features)
        return q_value

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs)
        self.action_dist = DirichletDistribution(1)

    def make_critics(self, observation_space: spaces.Space, action_space: spaces.Space) -> None:
        self.critic.qf1 = CustomCriticNetwork(observation_space, action_space, self.net_arch[0])
        self.critic.qf2 = CustomCriticNetwork(observation_space, action_space, self.net_arch[0])
        self.critic_target.qf1 = CustomCriticNetwork(observation_space, action_space, self.net_arch[0])
        self.critic_target.qf2 = CustomCriticNetwork(observation_space, action_space, self.net_arch[0])

    def _build(self, lr_schedule: Schedule) -> None:
        # Call the parent class's build method
        super(CustomSACPolicy, self)._build(lr_schedule)
  
        # Initialize critics
        self.make_critics(self.observation_space, self.action_space)
    
    def _get_action_dist_from_latent(self, latent_pi):
        concentration = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(concentration)

