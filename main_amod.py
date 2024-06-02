import torch
from src.envs.amod_env import AMoD
import os
import random
import gymnasium as gym
from enum import Enum

from torch.utils.tensorboard import SummaryWriter
from gymnasium.envs.registration import register
from src.algos.a2c_stable_baselines import CustomMultiInputActorCriticPolicy
from src.algos.sac_stable_baselines import CustomMultiInputSACPolicy
from src.envs.stable_baselines_env_wrapper import MyDummyVecEnv

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C, SAC, PPO

class RLAlgorithm(Enum):
    A2C = 0
    PPO = 1
    SAC = 2

RL_ALGORITHM = RLAlgorithm.SAC
CHECKPOINT_PATH = ""


random.seed(104)

writer = SummaryWriter()

# TODO: figure out GPU issue
#device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
device = "cpu"

class EvaluationCallback(BaseCallback):
    def __init__(self, eval_env, tensorboard_writer, eval_freq=1000, save_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.tensorboard_writer = tensorboard_writer
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_path = f"./amod_checkpoints/{RL_ALGORITHM.name}"
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 or self.num_timesteps == 1:
            validation_reward = self.evaluate_model()
            self.tensorboard_writer.add_scalar("Validation reward", validation_reward, self.num_timesteps)
            # print("Validation reward is ", validation_reward)
        if self.num_timesteps % self.save_freq == 0:
            self.save_checkpoint()
        return True
    
    def save_checkpoint(self):
        model_path = os.path.join(self.save_path, f"{self.num_timesteps}_steps.zip")
        self.model.save(model_path)
        print(f"Saving model checkpoint to {model_path}")

    def evaluate_model(self):
        # env.env_method("set_start_to_end_test", True)
        obs = self.eval_env.reset()
        eps_served_demand = 0
        eps_rebalancing_cost = 0
        eps_reward = 0
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            eps_served_demand += float(info[0]["served_demand"])
            eps_rebalancing_cost += float(info[0]["rebalancing_cost"])
            # we read reward from info instead of the returned value so that we can include the reward from
            # the first matching step, which happens in reset()
            eps_reward += float(info[0]["reward"])
        # Note that this is missing first step matching reward, since that step happens in reset()
        print(f"Reward: {eps_reward:.2f} | ServedDemand: {eps_served_demand:.2f} | Reb. Cost: {eps_rebalancing_cost:.2f}")
        return eps_reward


# Register the environment
register(id='CustomEnv-v0', entry_point=AMoD)

# Create the environment
env = MyDummyVecEnv([lambda: gym.make('CustomEnv-v0')])

policy_kwargs = dict(
    features_extractor_class=None,
    hidden_features_dim=256,
    node_features_dim=13,
    action_dim=1
)

if RL_ALGORITHM == RLAlgorithm.A2C:
    model = A2C(CustomMultiInputActorCriticPolicy,
                env, policy_kwargs=policy_kwargs, verbose=1,
                use_rms_prop=False, learning_rate=1e-3, ent_coef=0.3, device=device)
    eval_callback = EvaluationCallback(env, writer, eval_freq=1000, save_freq=10000)
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print("Loading saved model from path ", CHECKPOINT_PATH)
        model = A2C.load(CHECKPOINT_PATH, env=env, device=device)
elif RL_ALGORITHM == RLAlgorithm.PPO:
    model = PPO(CustomMultiInputActorCriticPolicy, env, policy_kwargs=policy_kwargs,
                verbose=1, learning_rate=1e-3, ent_coef=0.3, device=device)
    eval_callback = EvaluationCallback(env, writer, eval_freq=1000, save_freq=10000)
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print("Loading saved model from path ", CHECKPOINT_PATH)
        model = PPO.load(CHECKPOINT_PATH, env=env, device=device)
else:
    print("initializing SAC with device ", device)
    model = SAC(CustomMultiInputSACPolicy, env, policy_kwargs=policy_kwargs,
                verbose=1, learning_rate=1e-3, ent_coef=0.3, batch_size=100,
                gamma=0.99, learning_starts=10, device=device)
    eval_callback = EvaluationCallback(env, writer, eval_freq=100, save_freq=1000)
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print("Loading saved model from path ", CHECKPOINT_PATH)
        model = SAC.load(CHECKPOINT_PATH, env=env, device=device)

model.learn(total_timesteps=1000000000, callback=eval_callback)
