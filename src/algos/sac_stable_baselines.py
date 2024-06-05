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
from src.algos.stable_baselines_gcn import GCNCriticExtractor
from src.algos.stable_baselines_mpnn import MPNNCriticExtractor

    
class CustomContinuousCritic(ContinuousCritic):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.set_grad_enabled(True):
            # include actions as input to feature extractor
            obs_copy = obs.copy()
            obs_copy["node_features"] = torch.cat([obs["node_features"], actions.unsqueeze(-1)], dim=2)
            features = self.extract_features(obs_copy, self.features_extractor)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            # include actions as input to feature extractor
            obs_copy = obs.copy()
            obs_copy["node_features"] = torch.cat([obs["node_features"], actions.unsqueeze(-1)], dim=2)
            features = self.extract_features(obs_copy, self.features_extractor)
        return self.q_networks[0](torch.cat([features, actions], dim=1))

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs)
        self.action_dist = DirichletDistribution(1)

    # def make_critics(self, observation_space: spaces.Space, action_space: spaces.Space) -> None:
    #     self.critic.qf1 = CustomCriticNetwork(observation_space, action_space, self.net_arch[0])
    #     self.critic.qf2 = CustomCriticNetwork(observation_space, action_space, self.net_arch[0])
    #     self.critic_target.qf1 = CustomCriticNetwork(observation_space, action_space, self.net_arch[0])
    #     self.critic_target.qf2 = CustomCriticNetwork(observation_space, action_space, self.net_arch[0])

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )


        # self.critic = self.make_critic(features_extractor=GCNCriticExtractor(self.observation_space, self.features_extractor_kwargs['hidden_features_dim'],
        #                                                                      action_dim=self.action_space.shape[1]))
        self.critic = self.make_critic(features_extractor=MPNNCriticExtractor(self.observation_space, self.features_extractor_kwargs['hidden_features_dim'],
                                                                              action_dim=self.action_space.shape[1]))
        critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=GCNCriticExtractor(self.observation_space, self.features_extractor_kwargs['hidden_features_dim'],
                                                                                    action_dim=self.action_space.shape[1]))
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)
    
    def _get_action_dist_from_latent(self, latent_pi):
        concentration = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(concentration)

