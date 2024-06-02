from gymnasium import spaces
from typing import List, Tuple, Any, Dict, Optional, Union, Type, TypeVar
import torch.nn.functional as F
import torch
import torch.nn as nn

from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from src.algos.dirichlet_distribution import DirichletDistribution
from src.algos.stable_baselines_gcnn import GCNExtractor
from src.algos.stable_baselines_mpnn import MpnnExtractor


class CustomSACActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        hidden_features_dim: int = 256,
        node_features_dim: int = 13,
        edge_features_dim: int = 1,
        action_dim: int = 1,
        extractor_type: str = "gcn", # TODO: make this an enum
    ):
        super(CustomSACActor, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images
        )
        # don't include actions as input to network, so set action_dim = 0
        if extractor_type == "gcn":
            self.extractor = GCNExtractor(hidden_features_dim, node_features_dim, 0)
        elif extractor_type == "mpnn":
            self.extractor = MpnnExtractor(hidden_features_dim, node_features_dim, edge_features_dim, 0)
        self.action_dist = DirichletDistribution(action_dim)

    def get_action_dist_params(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            concentration
        """
        return self.extractor(obs)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        concentration = self.get_action_dist_params(obs)
        # TODO: move softplus here instead of in Dirichlet
        return_val = self.action_dist.actions_from_params(concentration, deterministic=deterministic) 
        return return_val

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
        concentration = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(concentration)

class CustomSACContinuousCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        hidden_features_dim: int = 256,
        node_features_dim: int = 13,
        edge_features_dim: int = 1,
        action_dim: int = 1,
        extractor_type = "gcn",
    ):
        super(CustomSACContinuousCritic, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor
        )
        self.extractors = []
        for idx in range(n_critics):
            if extractor_type == "gcn":
                extractor = GCNExtractor(hidden_features_dim, node_features_dim, action_dim)
            elif extractor_type == "mpnn":
                extractor = MpnnExtractor(hidden_features_dim, node_features_dim, edge_features_dim, action_dim)
            self.extractors.append(extractor)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.set_grad_enabled(True):
            return tuple(torch.sum(extractor(obs, actions), dim=1) for extractor in self.extractors)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sum(self.extractors[0](obs, actions), dim=1)

class CustomMultiInputSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        hidden_features_dim: int = 256,
        node_features_dim: int = 13,
        action_dim: int = 1,
        extractor_type: str = "gcn"
    ):
        self.squash_output = False
        self.hidden_features_dim = hidden_features_dim
        self.node_features_dim = node_features_dim
        self.action_dim = action_dim
        self.extractor_type = extractor_type
        super(CustomMultiInputSACPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor
        )

    @SACPolicy.squash_output.setter
    def squash_output(self, new_val: bool):
        self._squash_output = new_val

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        return None

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        # we are not using features extractor
        net_kwargs = net_kwargs.copy()
        net_kwargs.update(dict(features_extractor=features_extractor, features_dim=0))
        return net_kwargs
    
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomSACActor(**actor_kwargs, hidden_features_dim=self.hidden_features_dim,
                              node_features_dim=self.node_features_dim, action_dim=self.action_dim,
                              extractor_type=self.extractor_type).to(self.device)
    
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomSACContinuousCritic(**critic_kwargs, hidden_features_dim=self.hidden_features_dim,
                                         node_features_dim=self.node_features_dim, action_dim=self.action_dim,
                                         extractor_type=self.extractor_type).to(self.device)
