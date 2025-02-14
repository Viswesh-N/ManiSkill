from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional

import tqdm
from tqdm.auto import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import mani_skill.envs

###############################################################################
# 1. Arguments & Replay Buffer
###############################################################################

@dataclass
class Args:
    # Experiment and logging details
    exp_name: Optional[str] = None
    robot: str = "panda"
    control_mode: str = "pd_joint_delta_pos"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "maniskill_experiments"
    wandb_entity: str = "ucsd_erl"  # Set this to your valid wandb entity if available.
    wandb_group: str = "CQL"
    capture_video: bool = True
    save_trajectory: bool = False
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None
    log_freq: int = 1_000
    wandb_video_freq: int = 0
    save_model_dir: Optional[str] = "runs"

    # Environment specific arguments
    env_id: str = "PickCube-v1"
    env_vectorization: str = "gpu"
    num_envs: int = 16
    num_eval_envs: int = 16
    partial_reset: bool = False
    eval_partial_reset: bool = False
    num_steps: int = 50
    num_eval_steps: int = 50
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    eval_freq: int = 25
    save_train_video_freq: Optional[int] = None

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    gamma: float = 0.8
    tau: float = 0.01
    batch_size: int = 1024
    learning_starts: int = 4_000
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 1
    target_network_frequency: int = 1
    alpha: float = 0.2       # SAC entropy coefficient
    autotune: bool = True    # Automatic tuning of SAC alpha
    training_freq: int = 64
    utd: float = 0.5
    bootstrap_at_done: str = "always"
    
    # CQL-Specific Hyperparameters
    cql_alpha: float = 1.0
    """
    Weight of the CQL penalty term: 
    alpha * (logsumexp(Q) - Q(s, a_data)).
    """
    cql_n_actions: int = 10
    """
    Number of random actions to sample per state for the log-sum-exp approximation.
    """
    cql_autotune: bool = False
    cql_target_action_gap: float = 0.0
    """
    When auto-tuning the CQL penalty via a Lagrangian, we try to enforce:
        E[logsumexp(Q) - Q(s, a_data)] >= cql_target_action_gap
    """

    # To be computed at runtime
    grad_steps_per_iteration: int = 0
    steps_per_env: int = 0

@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs

        obs_shape = env.single_observation_space.shape
        act_shape = env.single_action_space.shape

        self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + obs_shape).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + obs_shape).to(storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs) + act_shape).to(storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)

    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )

###############################################################################
# 2. Networks: Q and Actor
###############################################################################

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_dim = int(np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self.net(x)

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_dim = int(np.prod(env.single_observation_space.shape))
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        act_dim = int(np.prod(env.single_action_space.shape))
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)

        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        # Rescale log_std into [LOG_STD_MIN, LOG_STD_MAX]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

###############################################################################
# 3. Logger
###############################################################################

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()

###############################################################################
# 4. Main Training Loop with CQL Updates
###############################################################################

if __name__ == "__main__":
    args = tyro.cli(Args)
    # Compute training and gradient update steps per iteration.
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs

    if args.exp_name is None:
        base = os.path.basename(__file__)
        args.exp_name = base[:-3] if base.endswith(".py") else base
        run_name = f"{args.env_id}__{args.exp_name}__{args.robot}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment Setup #######
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", robot_uids=args.robot)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    envs = gym.make(args.env_id,
                    num_envs=args.num_envs if not args.evaluate else 1,
                    reconfiguration_freq=args.reconfiguration_freq,
                    **env_kwargs)
    eval_envs = gym.make(args.env_id,
                         num_envs=args.num_eval_envs,
                         reconfiguration_freq=args.eval_reconfiguration_freq,
                         human_render_camera_configs=dict(shader_pack="default"),
                         **env_kwargs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # Video and trajectory recording
    if args.capture_video or args.save_trajectory:
        eval_output_dir = os.path.join(args.save_model_dir, run_name, "videos")
        if args.evaluate:
            eval_output_dir = os.path.join(os.path.dirname(args.checkpoint), "test_videos")
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=os.path.join(args.save_model_dir, run_name, "train_videos"),
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.save_trajectory,
            save_video=args.capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
            # wandb_video_freq=args.wandb_video_freq // args.eval_freq if args.eval_freq > 0 else 0
        )

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action spaces are supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # Logger / SummaryWriter
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs,
                                     num_envs=args.num_envs,
                                     env_id=args.env_id,
                                     reward_mode="normalized_dense",
                                     env_horizon=max_episode_steps,
                                     partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs,
                                          num_envs=args.num_eval_envs,
                                          env_id=args.env_id,
                                          reward_mode="normalized_dense",
                                          env_horizon=max_episode_steps,
                                          partial_reset=False)
            try:
                wandb.init(
                    project=args.wandb_project_name,
                    sync_tensorboard=False,
                    config=config,
                    name=run_name,
                    save_code=True,
                    group=args.wandb_group,
                    tags=["cql"]
                )
            except Exception as e:
                print("Wandb init error:", e)
                print("Retrying wandb init without entity parameter.")
                wandb.init(
                    project=args.wandb_project_name,
                    sync_tensorboard=False,
                    config=config,
                    name=run_name,
                    save_code=True,
                    group=args.wandb_group,
                    tags=["cql"]
                )
        writer = SummaryWriter(os.path.join(args.save_model_dir, run_name))
        writer.add_text("hyperparameters",
                        "|param|value|\n|-|-|\n%s" %
                        ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # Build networks
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)

    # Load checkpoint if provided
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt["actor"])
        qf1.load_state_dict(ckpt["qf1"])
        qf2.load_state_dict(ckpt["qf2"])  
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # SAC alpha auto-tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # CQL alpha auto-tuning (optional)
    if args.cql_autotune:
        cql_log_alpha = torch.zeros(1, requires_grad=True, device=device)
        cql_alpha = cql_log_alpha.exp().item()
        cql_a_optimizer = optim.Adam([cql_log_alpha], lr=args.q_lr)
    else:
        cql_alpha = args.cql_alpha

    # Create replay buffer
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device,
    )

    # Precompute action-space bounds for CQL random action sampling.
    action_dim = envs.single_action_space.shape[0]
    action_low = torch.as_tensor(envs.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(envs.single_action_space.high, device=device, dtype=torch.float32)

    # Initialize environment state
    obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    global_update = 0
    learning_has_started = False
    global_steps_per_iteration = args.num_envs * args.steps_per_env
    pbar = tqdm(total=args.total_timesteps, desc="Training CQL")
    cumulative_times = defaultdict(float)

    ###########################################################################
    # Main Loop
    ###########################################################################
    while global_step < args.total_timesteps:
        # Evaluation every eval_freq iterations (in env steps)
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_action = actor.get_eval_action(eval_obs)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum().item()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            pbar.set_description(f"success_once: {eval_metrics_mean.get('success_once', 0):.2f}, return: {eval_metrics_mean.get('return', 0):.2f}")
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
            actor.train()

            # Optionally save model checkpoint during training.
            if args.save_model:
                model_path = os.path.join(args.save_model_dir, run_name, f"ckpt_{global_step}.pt")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save({
                    "actor": actor.state_dict(),
                    "qf1": qf1_target.state_dict(),
                    "qf2": qf2_target.state_dict(),
                    "log_alpha": log_alpha if args.autotune else None,
                    "cql_log_alpha": cql_log_alpha if args.cql_autotune else None,
                }, model_path)
                print(f"model saved to {model_path}")

        # Rollout / data collection
        rollout_start = time.perf_counter()
        for _ in range(args.steps_per_env):
            global_step += args.num_envs
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _ = actor.get_action(obs)
                actions = actions.detach()
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()

            if args.bootstrap_at_done == "never":
                stop_bootstrap = terminations | truncations
            elif args.bootstrap_at_done == "always":
                stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
            else:
                stop_bootstrap = terminations

            if "final_info" in infos:
                done_mask = infos["_final_info"]
                real_next_obs[done_mask] = infos["final_observation"][done_mask]
                final_info = infos["final_info"]
                for k, v in final_info["episode"].items():
                    if logger is not None:
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_start
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # Do not start learning until sufficient data is collected.
        if global_step < args.learning_starts:
            continue

        # Training updates
        update_start = time.perf_counter()
        learning_has_started = True
        for _ in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            # 1. Standard TD target computation
            with torch.no_grad():
                next_action, next_log_pi, _ = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_action)
                qf2_next_target = qf2_target(data.next_obs, next_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi
                target_q = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

            # 2. Evaluate current Q values for data actions
            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)

            # 3. CQL log-sum-exp penalty.
            batch_size_ = data.obs.shape[0]
            rand_actions = torch.rand(batch_size_, args.cql_n_actions, action_dim, device=device)
            rand_actions = rand_actions * (action_high - action_low) + action_low

            obs_tiled = data.obs.unsqueeze(1).expand(batch_size_, args.cql_n_actions, -1)
            obs_tiled = obs_tiled.reshape(batch_size_ * args.cql_n_actions, -1)
            rand_actions_flat = rand_actions.reshape(batch_size_ * args.cql_n_actions, action_dim)

            qf1_rand = qf1(obs_tiled, rand_actions_flat).view(batch_size_, args.cql_n_actions)
            qf2_rand = qf2(obs_tiled, rand_actions_flat).view(batch_size_, args.cql_n_actions)

            qf1_rand_logsumexp = torch.logsumexp(qf1_rand, dim=1)
            qf2_rand_logsumexp = torch.logsumexp(qf2_rand, dim=1)

            cql_conservative_term_1 = qf1_rand_logsumexp.mean() - qf1_a_values.mean()
            cql_conservative_term_2 = qf2_rand_logsumexp.mean() - qf2_a_values.mean()

            if args.cql_autotune:
                cql_loss_1 = cql_alpha * (cql_conservative_term_1 - args.cql_target_action_gap)
                cql_loss_2 = cql_alpha * (cql_conservative_term_2 - args.cql_target_action_gap)
            else:
                cql_loss_1 = cql_alpha * cql_conservative_term_1
                cql_loss_2 = cql_alpha * cql_conservative_term_2

            # 4. Combine standard TD loss with CQL penalty.
            qf1_loss = F.mse_loss(qf1_a_values, target_q) + cql_loss_1
            qf2_loss = F.mse_loss(qf2_a_values, target_q) + cql_loss_2
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Optional: Lagrangian update for cql_alpha
            if args.cql_autotune:
                cql_alpha_loss = - (cql_log_alpha.exp() * (
                    (cql_conservative_term_1 - args.cql_target_action_gap).detach() +
                    (cql_conservative_term_2 - args.cql_target_action_gap).detach()
                ))
                cql_a_optimizer.zero_grad()
                cql_alpha_loss.backward()
                cql_a_optimizer.step()
                cql_alpha = cql_log_alpha.exp().item()

            # 5. Policy update (delayed)
            if global_update % args.policy_frequency == 0:
                pi, log_pi, _ = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi)
                qf2_pi = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = (alpha * log_pi - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # SAC alpha tuning
                if args.autotune:
                    with torch.no_grad():
                        _, log_pi_eval, _ = actor.get_action(data.obs)
                    alpha_loss = - (log_alpha.exp() * (log_pi_eval + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # 6. Soft-update the target networks.
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        update_time = time.perf_counter() - update_start
        cumulative_times["update_time"] += update_time

        # Logging training info every log_freq steps.
        if logger is not None and (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            logger.add_scalar("losses/qf_loss_total", qf_loss.item(), global_step)
            logger.add_scalar("losses/cql_term_1", cql_conservative_term_1.item(), global_step)
            logger.add_scalar("losses/cql_term_2", cql_conservative_term_2.item(), global_step)
            logger.add_scalar("losses/cql_loss_1", cql_loss_1.item(), global_step)
            logger.add_scalar("losses/cql_loss_2", cql_loss_2.item(), global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha", alpha, global_step)
            if args.cql_autotune:
                logger.add_scalar("losses/cql_alpha", cql_alpha, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            total_time = cumulative_times["rollout_time"] + cumulative_times["update_time"]
            logger.add_scalar("time/total_rollout+update_time", total_time, global_step)
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    # Final save
    if not args.evaluate and args.save_model:
        model_path = os.path.join(args.save_model_dir, run_name, "final_ckpt.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            "actor": actor.state_dict(),
            "qf1": qf1_target.state_dict(),
            "qf2": qf2_target.state_dict(),
            "log_alpha": log_alpha if args.autotune else None,
            "cql_log_alpha": cql_log_alpha if args.cql_autotune else None,
        }, model_path)
        print(f"model saved to {model_path}")
        if logger is not None:
            logger.close()

    envs.close()
    eval_envs.close()
