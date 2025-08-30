# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from model_selection_algorithms.algorithmsmodsel import BalancingClassic, BalancingHyperparamDoublingDataDriven, CorralHyperparam, EXP3Hyperparam, UCBHyperparam

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from utils.atari_wrappers import *
from utils.buffers import *


from utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from utils.buffers import ReplayBuffer


@dataclass
class Args:

    modsel_alg: str 
    env_id: str 
    seed: int = 0

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""

    modsel_options = ["D3RB", "ED2RB" ,"Corral", "UCB", "Exp3", "Classic"]
    hparam_to_tune: str = "architecture"
    num_base_learners: int = 3
    base_learners_hparam = [0, 1, 2]      # index of architecture
    # num_base_learners: int = 1
    # base_learners_hparam = [1]      # index of architecture
   
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Modsel_DQN_Architecture"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str =  "BreakoutNoFrameskip-v4" # "PhoenixNoFrameskip-v4" 
    """the id of the environment"""
    architecture: int = 1
    "the network architecture"
    total_timesteps: int = int(10e6)
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100000 #1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


class BaseLearner(nn.Module):
    def __init__(self, base_index, envs, device, args):
        super().__init__()
        self.base_index = base_index
        self.internal_time = 0

        
        self.q_network = QNetwork(envs, args.base_learners_hparam[base_index]).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.target_network = QNetwork(envs, args.base_learners_hparam[base_index]).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        ) 
        
    
        
    def increment_internal_time(self, t):
        self.internal_time += t 

    def get_internal_time(self):
        return self.internal_time


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, architecture):
        super().__init__()
        self.network = self.set_network(architecture, env)


    def set_network(self, index, env):
        network = None
        if index == -1:
            network = nn.Sequential(        # Shallow network: only one convolutional layer, limits the ability to learn hierarchical features
                nn.Conv2d(4, 16, 8, stride=4),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(6400, env.single_action_space.n),
            )

        elif index == 0:
            network = nn.Sequential(        # Shallow network: only one convolutional layer, limits the ability to learn hierarchical features
                nn.Conv2d(4, 16, 8, stride=4),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(6400,256),
                nn.ReLU(),
                nn.Linear(256, env.single_action_space.n),
            )

        elif index == 1:
            network = nn.Sequential(
                    nn.Conv2d(4, 32, 8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(3136, 512),
                    nn.ReLU(),
                    nn.Linear(512, env.single_action_space.n),
                )

        elif index == 2:
            network = nn.Sequential(        # Narrow network: Small hidden layer size and smaller number of channels
                nn.Conv2d(4, 8, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(8, 8, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(8, 8, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(392, 32),
                nn.ReLU(),
                nn.Linear(32, env.single_action_space.n),
            )
        
        return network


    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def normalize_episodic_return(episodic_return, normalizer_const, args):
    if args.env_id == "PongNoFrameskip-v4":
        normalized_return = (episodic_return + 21) / 42
    elif args.env_id == "BreakoutNoFrameskip-v4":
        normalized_return = (episodic_return / 450)
    elif args.env_id == "BoxingNoFrameskip-v4":
        normalized_return = (episodic_return + 100) / 200
    else:
        if episodic_return < normalizer_const:
            normalized_return = episodic_return/normalizer_const
        else:
            normalized_return = 1
    return normalized_return


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.modsel_alg}__{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

     # base learners initiation
    m = args.num_base_learners
    base_learners = []
    for i in range(m):
        agent = BaseLearner(base_index=i, envs=envs, device=device, args=args).to(device)
        base_learners.append(agent)

    # meta learner initiation
    if args.modsel_alg == "D3RB" :
        modsel = BalancingHyperparamDoublingDataDriven(m, dmin = 1)
    elif args.modsel_alg == "ED2RB" :
        modsel = BalancingHyperparamDoublingDataDriven(m, dmin = 1, empirical = True)
    elif args.modsel_alg == "Corral":
        modsel = CorralHyperparam(m)
    elif args.modsel_alg == "Exp3":
        modsel = EXP3Hyperparam(m)
    elif args.modsel_alg == "UCB":
        modsel = UCBHyperparam(m)
    elif args.modsel_alg == "Classic":
        putative_bounds_multipliers = [1]*m
        modsel = BalancingClassic(m, putative_bounds_multipliers)

    selected_base_learners = []


    # Sample the first base learner
    base_index = modsel.sample_base_index()
    selected_base_learners.append(base_index)
    agent = base_learners[base_index]
    optimizer = agent.optimizer
    q_network = agent.q_network
    target_network = agent.target_network
    rb = agent.rb


    start_time = time.time()
    
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # MODSEL: Sample new base agnet 
        if global_step>0 and "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    base_index = modsel.sample_base_index()
                    selected_base_learners.append(base_index)
                    agent = base_learners[base_index]
                    optimizer = agent.optimizer
                    q_network = agent.q_network
                    target_network = agent.target_network
                    rb = agent.rb

                    
        agent_step = agent.get_internal_time()

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().detach().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                    # MODSEL: Update current base learner ß
                    episodic_return = info['episode']['r']
                    normalized_return = normalize_episodic_return(episodic_return, normalizer_const=15000, args=args)
                    modsel.update_distribution(base_index, normalized_return)
                    
                    writer.add_scalar("modelselection/MetaLearner", normalized_return, global_step)
                    writer.add_scalar(f"baselearners/architecture={args.base_learners_hparam[base_index]}", normalized_return, global_step)
                    

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
       
        if agent_step > args.learning_starts:
            if agent_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar(f"losses/td_loss_{base_index}", loss, global_step)
                    writer.add_scalar(f"losses/q_values_{base_index}", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    writer.add_scalar("modelselection/selected_base_learner", args.base_learners_hparam[base_index], global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if agent_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

        agent.increment_internal_time(1)


    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        

        
    envs.close()
    writer.close()
