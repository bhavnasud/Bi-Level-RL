import os
import random
import gymnasium as gym
import argparse
import torch

from src.envs.amod_env import AMoD, Scenario
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C, SAC, PPO
from torch.utils.tensorboard import SummaryWriter
from datetime import date
from gymnasium.envs.registration import register
from src.algos.a2c_stable_baselines import CustomMultiInputActorCriticPolicy
from src.algos.sac_stable_baselines import CustomSACPolicy
from src.envs.stable_baselines_env_wrapper import MyDummyVecEnv
from src.misc.utils import FeatureExtractor, RLAlgorithm
from src.algos.stable_baselines_gcn import GCNExtractor
from src.algos.stable_baselines_mpnn import MPNNExtractor
from src.algos.stable_baselines_mlp import MLPExtractor

parser = argparse.ArgumentParser(description='A2C-GNN')

# Simulator parameters
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.8, metavar='S',
                    help='demand_ratio (default: 0.8)')
parser.add_argument('--time_start', type=int, default=7, metavar='S',
                    help='simulation start time in hours (default: 7 hr)')
parser.add_argument('--duration', type=int, default=2, metavar='S',
                    help='episode duration in hours (default: 2 hr)')
parser.add_argument('--time_horizon', type=int, default=10, metavar='S',
                    help='matching steps in the future for demand and arriving vehicle forecast (default: 10 min)')
parser.add_argument('--matching_tstep', type=int, default=1, metavar='S',
                    help='minutes per timestep (default: 1 min)')
parser.add_argument('--max_waiting_time', type=int, default=10, metavar='S',
                    help='maximum passengers waiting time for a ride (default: 10 min)')
parser.add_argument('--beta', type=float, default=1, metavar='S',   ########## MODIFIED THE DEFUAULT VALUE
                    help='cost of rebalancing (default: 1)')
parser.add_argument('--num_regions', type=int, default=8, metavar='S',
                    help='Number of regions fro spatial aggregation (default: 8)')
parser.add_argument('--acc_init', type=int, default=90, metavar='S',
                    help='Initial number of taxis per region (default: 90)')
# Model parameters
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--agent_name', type=str, default=date.today().strftime("%Y%m%d")+'_a2c_gnn',
                    help='Pretrained agent selection for agent evaluation (default:'+date.today().strftime("%Y%m%d")+'_a2c_gnn')
parser.add_argument('--cplexpath', type=str, default='/opt/ibm/ILOG/CPLEX_Studio2211/opl/bin/x86-64_linux/',
                    help='defines directory of the CPLEX installation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=5000, metavar='N',
                    help='number of episodes to train agent (default: 16k)')
parser.add_argument('--max_steps', type=int, default=120, metavar='N',
                    help='number of steps per episode (default: T=120)')
parser.add_argument('--no-cuda', type=bool, default=True,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# set this to the path to the saved checkpoint (e.g. amod_checkpoints/SAC/GCN/100000_steps.zip) to resume
# from that checkpoint
CHECKPOINT_PATH = ""

random.seed(args.seed)

writer = SummaryWriter()


class EvaluationCallback(BaseCallback):
    def __init__(self, eval_env, tensorboard_writer, eval_freq=1000, save_freq=10000, verbose=1, rl_algorithm=RLAlgorithm.SAC,
                 feature_extractor=FeatureExtractor.GCN):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.tensorboard_writer = tensorboard_writer
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_path = f"./amod_checkpoints/{rl_algorithm.name}/{feature_extractor.name}/"
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 or self.num_timesteps == 1:
            validation_reward = self.evaluate_model()
            self.tensorboard_writer.add_scalar("Validation reward", validation_reward, self.num_timesteps)
        if self.num_timesteps % self.save_freq == 0:
            self.save_checkpoint()
        return True
    
    def save_checkpoint(self):
        model_path = os.path.join(self.save_path, f"{self.num_timesteps}_steps.zip")
        self.model.save(model_path)
        print(f"Saving model checkpoint to {model_path}")

    def evaluate_model(self):
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
        print(f"Reward: {eps_reward:.2f} | ServedDemand: {eps_served_demand:.2f} | Reb. Cost: {eps_rebalancing_cost:.2f}")
        return eps_reward


def run_training(feature_extractor, rl_algorithm, args):
    run_dir = os.path.join('amod_runs', f'{rl_algorithm.name}/{feature_extractor.name}')
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    total_steps = args.max_episodes * args.duration * 60

    # Register the environment
    register(id='CustomEnv-v0', entry_point=AMoD, kwargs={'args': args, 'beta': 1, 'city': 'lux'})

    # Create the environment
    env = MyDummyVecEnv([lambda: gym.make('CustomEnv-v0')])

    if feature_extractor == FeatureExtractor.GCN:
        features_extractor_class = GCNExtractor
    elif feature_extractor == FeatureExtractor.MPNN:
        features_extractor_class = MPNNExtractor
    else:
        features_extractor_class = MLPExtractor

    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs={
            "hidden_features_dim": 256, # TODO: make this a cmdline argument
            "num_nodes": env.envs[0].nregions
        }
    )

    if rl_algorithm == RLAlgorithm.A2C:
        model = A2C(CustomMultiInputActorCriticPolicy,
                    env, policy_kwargs=policy_kwargs, verbose=1,
                    use_rms_prop=False, learning_rate=1e-3, ent_coef=0.3, n_steps=1,
                    gamma=0.99, device=device, tensorboard_log='/tensorboard_log/')

        eval_callback = EvaluationCallback(env, writer, eval_freq=1000, save_freq=10000,
                                           rl_algorithm=rl_algorithm, feature_extractor=feature_extractor)
        if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            print("Loading saved model from path ", CHECKPOINT_PATH)
            model = A2C.load(CHECKPOINT_PATH, env=env, device=device)
    elif rl_algorithm == RLAlgorithm.PPO:
        model = PPO(CustomMultiInputActorCriticPolicy, env, policy_kwargs=policy_kwargs,
                    verbose=1, learning_rate=1e-3, ent_coef=0.3, n_steps=1,
                   gamma=0.99, device=device, tensorboard_log='/tensorboard_log/')
        eval_callback = EvaluationCallback(env, writer, eval_freq=1000, save_freq=10000,
                                           rl_algorithm=rl_algorithm, feature_extractor=feature_extractor)
        if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            print("Loading saved model from path ", CHECKPOINT_PATH)
            model = PPO.load(CHECKPOINT_PATH, env=env, device=device)
    else:
        model = SAC(CustomSACPolicy, env, policy_kwargs=policy_kwargs,
                verbose=1, learning_rate=1e-3, ent_coef=0.3, batch_size=100,
                gamma=0.99, learning_starts=10, device=device, seed=1, tensorboard_log='/tensorboard_log/')
        eval_callback = EvaluationCallback(env, writer, eval_freq=100, save_freq=1000,
                                           rl_algorithm=rl_algorithm, feature_extractor=feature_extractor)
        if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            print("Loading saved model from path ", CHECKPOINT_PATH)
            model = SAC.load(CHECKPOINT_PATH, env=env, device=device)

    model.learn(total_timesteps=total_steps, callback=eval_callback)


# note that a MPNN won't work here because the amod observation space does not have an edge attr
run_training(FeatureExtractor.GCN, RLAlgorithm.A2C, args)
