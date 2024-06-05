# import gymnasium as gym
# from typing import List, Tuple, Any, Callable, Dict, Optional, Union, Type
# import torch
# import torch.nn as nn
# from src.algos.dirichlet_distribution import DirichletDistribution


# from stable_baselines3.common.type_aliases import  Schedule
# from stable_baselines3.common.distributions import Distribution
# from stable_baselines3.common.policies import MultiInputActorCriticPolicy
# from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
# from src.misc.utils import get_rl_network, FeatureExtractor


# class CustomMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):
#     def __init__(
#         self,
#         observation_space: gym.spaces.Space,
#         action_space: gym.spaces.Space,
#         lr_schedule: Callable[[float], float],
#         net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
#         activation_fn: Type[nn.Module] = nn.Tanh,
#         hidden_features_dim: int = 256,
#         node_features_dim: int = 13,
#         edge_features_dim: int = 1,
#         num_nodes: int = 1,
#         action_dim: int = 1, # number of actions per node
#         extractor_type=FeatureExtractor.MPNN,
#         *args,
#         **kwargs,
#     ):
#         self.extractor_type = extractor_type
#         self.hidden_features_dim = hidden_features_dim
#         self.node_features_dim = node_features_dim
#         self.edge_features_dim = edge_features_dim
#         self.action_dim = action_dim
#         self.num_nodes = num_nodes
#         super(CustomMultiInputActorCriticPolicy, self).__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             # Pass remaining arguments to base class
#             *args,
#             **kwargs,
#         )
#         # Disable orthogonal initialization
#         self.ortho_init = False

#     def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Forward pass in all the networks (actor and critic)

#         :param obs: Observation
#         :param deterministic: Whether to sample or use deterministic actions
#         :return: action, value and log probability of the action
#         """
#         distribution = self.get_distribution(obs)
#         actions = distribution.get_actions(deterministic=deterministic)
#         log_probs = distribution.log_prob(actions)
#         values = self.predict_values(obs)
#         return actions, log_probs, values


#     def _build(self, lr_schedule: Schedule) -> None:
#         """
#         Create the networks and the optimizer.

#         :param lr_schedule: Learning rate schedule
#             lr_schedule(1) is the initial learning rate
#         """
#         self.actor_network = get_rl_network(extractor_type=self.extractor_type, hidden_features_dim=self.hidden_features_dim,
#                                                       node_features_dim=self.node_features_dim, edge_features_dim=self.edge_features_dim,
#                                                       num_nodes=self.num_nodes, action_dim=self.action_dim)
#         self.critic_network = get_rl_network(extractor_type=self.extractor_type, hidden_features_dim=self.hidden_features_dim,
#                                                       node_features_dim=self.node_features_dim, edge_features_dim=self.edge_features_dim,
#                                                       num_nodes=self.num_nodes, action_dim=self.action_dim, value_net=True, value_net_include_actions=False)
#         self.action_dist = DirichletDistribution(self.action_dim)

#         # Setup optimizer with initial learning rate
#         self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
    
#     def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
#         """
#         Evaluate actions according to the current policy,
#         given the observations.

#         :param obs: Observation
#         :param actions: Actions
#         :return: estimated value, log likelihood of taking those actions
#             and entropy of the action distribution.
#         """
#         distribution = self.get_distribution(obs)
#         log_probs = distribution.log_prob(actions)
#         values = self.predict_values(obs)
#         entropy = distribution.entropy()
#         return values, log_probs, entropy
    
#     def get_distribution(self, obs: PyTorchObs) -> Distribution:
#         """
#         Get the current policy distribution given the observations.

#         :param obs:
#         :return: the action distribution.
#         """
#         actor_concentration = self.actor_network(obs).squeeze(2)
#         return self.action_dist.proba_distribution(actor_concentration)
    
#     def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
#         """
#         Get the estimated values according to the current policy given the observations.

#         :param obs: Observation
#         :return: the estimated values.
#         """
#         return self.critic_network(obs)



import torch
from src.algos.dirichlet_distribution import DirichletDistribution
from stable_baselines3.common.type_aliases import Schedule

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

# class CustomNetwork(nn.Module):
#     """
#     Custom network for policy and value function.
#     It receives as input the features extracted by the feature extractor.

#     :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
#     :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
#     :param last_layer_dim_vf: (int) number of units for the last layer of the value network
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#     ):
#         super(CustomNetwork, self).__init__()

#         # Policy network
#         self.policy_net = nn.Sequential(
#             nn.Linear(feature_dim, 1)
#         )
#         # Value network
#         self.value_net = nn.Sequential(
#             nn.Linear(feature_dim, 1)
#         )

#     def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
#         a_probs = self.policy_net(features).squeeze(2)
#         return a_probs
    
#     def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
#         features = torch.sum(features, dim=1)
#         return_val = self.value_net(features)
#         return return_val
    


class CustomMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        *args,
        **kwargs
    ):

        super(CustomMultiInputActorCriticPolicy, self).__init__(
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.vf_features_extractor = GCNCriticExtractor(self.observation_space, **self.features_extractor_kwargs)


    # def _build_mlp_extractor(self) -> None:
    #     self.mlp_extractor = CustomNetwork(self.features_dim)
    
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()
        action_dim = self.action_space.shape[-1]
        self.action_dist = DirichletDistribution(action_dim)
        self.action_net = self.action_dist.proba_distribution_net(0)
        # self.value_net = nn.Identity()

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits)