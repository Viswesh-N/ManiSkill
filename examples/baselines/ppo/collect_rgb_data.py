#!/usr/bin/env python3
"""
Enhanced RGB Data Collection Script for ManiSkill Environments with Rewards

This script collects 128x128 RGB images, segmentation masks, actions, rewards, 
and episode information from a ManiSkill environment for RL training data.

Saves complete (s, a, r, s', done) tuples plus episode metadata.
"""

import os
import json
import time
import random
import argparse
from collections import defaultdict
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from PIL import Image
from tqdm import tqdm

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@dataclass
class CollectionArgs:
    target_dir: str
    """Directory to save collected data"""
    env_id: str = "PickCube-v1"
    """Environment ID to collect data from"""
    total_steps: int = 100000
    """Total number of steps to collect data for"""
    num_envs: int = 8
    """Number of parallel environments"""
    control_mode: str = "pd_joint_delta_pos"
    """Control mode for the environment"""
    seed: int = 42
    """Random seed"""
    image_size: int = 128
    """Size of RGB images (will be image_size x image_size)"""
    action_mode: str = "random"
    """Action selection mode: 'random' or 'trained' (if you have a trained model)"""
    save_frequency: int = 1
    """Save every N steps (1 means save every step)"""
    reward_mode: str = "dense"
    """Reward mode: 'sparse', 'dense', 'normalized_dense', or 'none'"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()
        extractors = {}
        self.out_features = 0
        feature_size = 256

        # Get the first camera's data to determine shapes
        sensor_data = sample_obs["sensor_data"]
        cam_name = list(sensor_data.keys())[0]
        cam_data = sensor_data[cam_name]

        # RGB processing
        if "rgb" in cam_data:
            in_channels = cam_data["rgb"].shape[-1]
            cnn = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            with torch.no_grad():
                n_flatten = cnn(cam_data["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size

        # Segmentation processing (if available)
        if "segmentation" in cam_data:
            seg_channels = cam_data["segmentation"].shape[-1]
            seg_cnn = nn.Sequential(
                nn.Conv2d(in_channels=seg_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            with torch.no_grad():
                n_flatten_seg = seg_cnn(cam_data["segmentation"].float().permute(0, 3, 1, 2).cpu()).shape[1]
                seg_fc = nn.Sequential(nn.Linear(n_flatten_seg, feature_size), nn.ReLU())
            extractors["segmentation"] = nn.Sequential(seg_cnn, seg_fc)
            self.out_features += feature_size

        # State processing (if available)
        if "extra" in sample_obs and "state" in sample_obs["extra"]:
            state_size = sample_obs["extra"]["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations):
        encoded_tensor_list = []
        
        # Get sensor data
        sensor_data = observations["sensor_data"]
        cam_name = list(sensor_data.keys())[0]
        cam_data = sensor_data[cam_name]
        
        # Process each modality
        for key, extractor in self.extractors.items():
            if key == "rgb" and "rgb" in cam_data:
                obs = cam_data["rgb"].float().permute(0, 3, 1, 2) / 255
                encoded_tensor_list.append(extractor(obs))
            elif key == "segmentation" and "segmentation" in cam_data:
                obs = cam_data["segmentation"].float().permute(0, 3, 1, 2)
                # Normalize segmentation values
                obs = obs / (obs.max() + 1e-8)  # Avoid division by zero
                encoded_tensor_list.append(extractor(obs))
            elif key == "state" and "extra" in observations and "state" in observations["extra"]:
                obs = observations["extra"]["state"]
                encoded_tensor_list.append(extractor(obs))
        
        if encoded_tensor_list:
            return torch.cat(encoded_tensor_list, dim=1)
        else:
            # Fallback if no features are extracted
            batch_size = list(sensor_data.values())[0]["rgb"].shape[0] if sensor_data else 1
            return torch.zeros((batch_size, 256), device=list(sensor_data.values())[0]["rgb"].device if sensor_data else torch.device('cpu'))


class SimpleAgent(nn.Module):
    """Simple agent that can take random actions or use a trained policy"""
    
    def __init__(self, envs, sample_obs, mode="random"):
        super().__init__()
        self.mode = mode
        self.action_space = envs.single_action_space
        
        if mode == "trained":
            self.feature_net = NatureCNN(sample_obs=sample_obs)
            latent_size = self.feature_net.out_features
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(latent_size, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01),
            )
    
    def get_action(self, obs):
        if self.mode == "random":
            # Random actions - determine batch size from sensor data
            sensor_data = obs["sensor_data"]
            cam_name = list(sensor_data.keys())[0]
            batch_size = sensor_data[cam_name]["rgb"].shape[0]
            
            actions = []
            for _ in range(batch_size):
                action = self.action_space.sample()
                actions.append(action)
            return torch.tensor(np.array(actions), dtype=torch.float32)
        else:
            # Use trained policy
            features = self.feature_net(obs)
            return self.actor_mean(features)


class EnhancedDataCollector:
    def __init__(self, args: CollectionArgs):
        self.args = args
        self.setup_directories()
        self.setup_environment()
        self.setup_agent()
        self.metadata = {
            "images": [],
            "next_images": [],
            "segmentation_masks": [],
            "next_segmentation_masks": [],
            "actions": [],
            "rewards": [],
            "terminations": [],
            "truncations": [],
            "episode_info": [],
            "env_info": {
                "env_id": args.env_id,
                "control_mode": args.control_mode,
                "reward_mode": args.reward_mode,
                "image_size": args.image_size,
                "total_steps": args.total_steps,
                "num_envs": args.num_envs,
            }
        }
        
        # Episode tracking
        self.episode_rewards = [0.0] * args.num_envs
        self.episode_lengths = [0] * args.num_envs
        self.episode_count = 0
        
    def setup_directories(self):
        """Create organized directory structure"""
        self.base_dir = self.args.target_dir
        self.images_dir = os.path.join(self.base_dir, "images")
        self.next_images_dir = os.path.join(self.base_dir, "next_images")
        self.segmentation_dir = os.path.join(self.base_dir, "segmentation")
        self.next_segmentation_dir = os.path.join(self.base_dir, "next_segmentation")
        self.metadata_dir = os.path.join(self.base_dir, "metadata")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.next_images_dir, exist_ok=True)
        os.makedirs(self.segmentation_dir, exist_ok=True)
        os.makedirs(self.next_segmentation_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        print(f"Data will be saved to: {self.base_dir}")
        print(f"Images directory: {self.images_dir}")
        print(f"Next images directory: {self.next_images_dir}")
        print(f"Segmentation directory: {self.segmentation_dir}")
        print(f"Next segmentation directory: {self.next_segmentation_dir}")
        print(f"Metadata directory: {self.metadata_dir}")

    def setup_environment(self):
        """Setup ManiSkill environment with RGB and segmentation observations"""
        # Set seeds
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        
        # Environment setup with RGB and segmentation mode
        env_kwargs = {
            "obs_mode": "rgb+segmentation",
            "render_mode": "rgb_array",
            "sim_backend": "physx_cuda",
            "control_mode": self.args.control_mode,
            "reward_mode": self.args.reward_mode,  # Explicitly set reward mode
        }
        
        # Create environment
        self.envs = gym.make(
            self.args.env_id,
            num_envs=self.args.num_envs,
            **env_kwargs
        )
        
        # Wrap with action space flattening if needed
        if isinstance(self.envs.action_space, gym.spaces.Dict):
            self.envs = FlattenActionSpaceWrapper(self.envs)
            
        # Wrap with ManiSkill vector environment
        self.envs = ManiSkillVectorEnv(self.envs, self.args.num_envs, ignore_terminations=True)
        
        # Get segmentation ID map for reference
        self.segmentation_id_map = self.envs._env.unwrapped.segmentation_id_map
        
        print(f"Environment setup complete: {self.args.env_id}")
        print(f"Reward mode: {self.args.reward_mode}")
        print(f"Action space: {self.envs.single_action_space}")
        print(f"Observation space: {self.envs.single_observation_space}")
        
        # Print segmentation information
        print("\nSegmentation ID Map:")
        for obj_id, obj in sorted(self.segmentation_id_map.items()):
            if hasattr(obj, '__class__'):
                print(f"  {obj_id}: {obj.__class__.__name__}, name - {getattr(obj, 'name', 'N/A')}")

    def setup_agent(self):
        """Setup agent for action generation"""
        # Reset environment to get sample observation
        obs, _ = self.envs.reset(seed=self.args.seed)
        self.agent = SimpleAgent(self.envs, sample_obs=obs, mode=self.args.action_mode)
        print(f"Agent setup complete in {self.args.action_mode} mode")

    def resize_image(self, image_array):
        """Resize image to target size"""
        if image_array.shape[:2] != (self.args.image_size, self.args.image_size):
            image = Image.fromarray(image_array.astype(np.uint8) if len(image_array.shape) == 3 else image_array.astype(np.uint16))
            image = image.resize((self.args.image_size, self.args.image_size), 
                               Image.LANCZOS if len(image_array.shape) == 3 else Image.NEAREST)
            image_array = np.array(image)
        return image_array

    def save_step_data(self, step: int, env_idx: int, 
                      current_rgb: np.ndarray, current_seg: np.ndarray,
                      next_rgb: np.ndarray, next_seg: np.ndarray,
                      action: np.ndarray, reward: float, 
                      terminated: bool, truncated: bool, info: dict):
        """Save complete (s, a, r, s', done) tuple for a single step"""
        
        # Create filenames
        current_rgb_filename = f"step_{step:08d}_env_{env_idx:02d}_rgb.png"
        current_seg_filename = f"step_{step:08d}_env_{env_idx:02d}_seg.png"
        next_rgb_filename = f"step_{step:08d}_env_{env_idx:02d}_next_rgb.png"
        next_seg_filename = f"step_{step:08d}_env_{env_idx:02d}_next_seg.png"
        
        current_rgb_path = os.path.join(self.images_dir, current_rgb_filename)
        current_seg_path = os.path.join(self.segmentation_dir, current_seg_filename)
        next_rgb_path = os.path.join(self.next_images_dir, next_rgb_filename)
        next_seg_path = os.path.join(self.next_segmentation_dir, next_seg_filename)
        
        # Resize and save images
        resized_current_rgb = self.resize_image(current_rgb)
        resized_current_seg = self.resize_image(current_seg)
        resized_next_rgb = self.resize_image(next_rgb)
        resized_next_seg = self.resize_image(next_seg)
        
        Image.fromarray(resized_current_rgb.astype(np.uint8)).save(current_rgb_path)
        Image.fromarray(resized_current_seg.astype(np.uint16)).save(current_seg_path)
        Image.fromarray(resized_next_rgb.astype(np.uint8)).save(next_rgb_path)
        Image.fromarray(resized_next_seg.astype(np.uint16)).save(next_seg_path)
        
        # Add to metadata
        current_rgb_relative = os.path.join("images", current_rgb_filename)
        current_seg_relative = os.path.join("segmentation", current_seg_filename)
        next_rgb_relative = os.path.join("next_images", next_rgb_filename)
        next_seg_relative = os.path.join("next_segmentation", next_seg_filename)
        
        self.metadata["images"].append(current_rgb_relative)
        self.metadata["segmentation_masks"].append(current_seg_relative)
        self.metadata["next_images"].append(next_rgb_relative)
        self.metadata["next_segmentation_masks"].append(next_seg_relative)
        self.metadata["actions"].append(action.tolist())
        self.metadata["rewards"].append(float(reward))
        self.metadata["terminations"].append(bool(terminated))
        self.metadata["truncations"].append(bool(truncated))
        
        # Episode info
        episode_info = {
            "step": step,
            "env_idx": env_idx,
            "episode_reward": self.episode_rewards[env_idx],
            "episode_length": self.episode_lengths[env_idx],
            "success": info.get("success", {}).get(env_idx, False) if isinstance(info.get("success", {}), dict) else bool(info.get("success", [False] * self.args.num_envs)[env_idx]),
            "fail": info.get("fail", {}).get(env_idx, False) if isinstance(info.get("fail", {}), dict) else bool(info.get("fail", [False] * self.args.num_envs)[env_idx]),
        }
        self.metadata["episode_info"].append(episode_info)
        
        return current_rgb_relative, current_seg_relative, next_rgb_relative, next_seg_relative

    def collect_data(self):
        """Main data collection loop"""
        print(f"Starting enhanced data collection for {self.args.total_steps} steps...")
        
        obs, _ = self.envs.reset(seed=self.args.seed)
        step_count = 0
        saved_count = 0
        
        pbar = tqdm(total=self.args.total_steps, desc="Collecting data")
        
        try:
            while step_count < self.args.total_steps:
                # Get current state data
                current_sensor_data = obs["sensor_data"]
                cam_name = list(current_sensor_data.keys())[0]
                
                # Get actions from agent
                with torch.no_grad():
                    actions = self.agent.get_action(obs)
                
                # Step environment
                next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
                
                # Get next state data
                next_sensor_data = next_obs["sensor_data"]
                
                # Update episode tracking
                for env_idx in range(self.args.num_envs):
                    self.episode_rewards[env_idx] += float(rewards[env_idx])
                    self.episode_lengths[env_idx] += 1
                    
                    # Reset episode tracking if episode ended
                    if terminations[env_idx] or truncations[env_idx]:
                        self.episode_rewards[env_idx] = 0.0
                        self.episode_lengths[env_idx] = 0
                        self.episode_count += 1
                
                # Save data if it's time
                if step_count % self.args.save_frequency == 0:
                    for env_idx in range(self.args.num_envs):
                        if step_count >= self.args.total_steps:
                            break
                            
                        # Extract current state
                        current_rgb = current_sensor_data[cam_name]["rgb"][env_idx].cpu().numpy()
                        current_seg = current_sensor_data[cam_name]["segmentation"][env_idx].cpu().numpy()
                        if len(current_seg.shape) == 3 and current_seg.shape[-1] == 1:
                            current_seg = current_seg.squeeze(-1)
                        
                        # Extract next state
                        next_rgb = next_sensor_data[cam_name]["rgb"][env_idx].cpu().numpy()
                        next_seg = next_sensor_data[cam_name]["segmentation"][env_idx].cpu().numpy()
                        if len(next_seg.shape) == 3 and next_seg.shape[-1] == 1:
                            next_seg = next_seg.squeeze(-1)
                        
                        action = actions[env_idx].cpu().numpy()
                        reward = rewards[env_idx].cpu().numpy()
                        terminated = terminations[env_idx].cpu().numpy()
                        truncated = truncations[env_idx].cpu().numpy()
                        
                        # Save the complete (s, a, r, s', done) tuple
                        self.save_step_data(
                            step_count, env_idx, 
                            current_rgb, current_seg,
                            next_rgb, next_seg,
                            action, reward, terminated, truncated, infos
                        )
                        saved_count += 1
                        step_count += 1
                        pbar.update(1)
                
                obs = next_obs
                
                # Save metadata periodically
                if saved_count % 1000 == 0:
                    self.save_metadata()
                    
        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        
        pbar.close()
        
        # Final metadata save
        self.save_metadata()
        print(f"\nData collection complete!")
        print(f"Total transitions saved: {saved_count}")
        print(f"Total episodes completed: {self.episode_count}")
        print(f"Data saved to: {self.base_dir}")

    def save_metadata(self):
        """Save metadata to JSON file"""
        metadata_path = os.path.join(self.metadata_dir, "enhanced_collection_metadata.json")
        
        # Add collection info
        self.metadata["collection_info"] = {
            "total_transitions": len(self.metadata["images"]),
            "total_episodes": self.episode_count,
            "collection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": vars(self.args)
        }
        
        # Add segmentation ID map for reference
        segmentation_map = {}
        for obj_id, obj in self.segmentation_id_map.items():
            segmentation_map[int(obj_id)] = {
                "class_name": obj.__class__.__name__ if hasattr(obj, '__class__') else "Unknown",
                "name": getattr(obj, 'name', 'N/A')
            }
        self.metadata["segmentation_id_map"] = segmentation_map
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect enhanced RGB data with rewards from ManiSkill environment")
    parser.add_argument("target_dir", type=str, help="Directory to save collected data")
    parser.add_argument("--env_id", type=str, default="PickCube-v1", help="Environment ID")
    parser.add_argument("--total_steps", type=int, default=1000, help="Total steps to collect")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos", help="Control mode")
    parser.add_argument("--reward_mode", type=str, default="dense", choices=["sparse", "dense", "normalized_dense", "none"], help="Reward mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image_size", type=int, default=128, help="Image size (width and height)")
    parser.add_argument("--action_mode", type=str, default="random", choices=["random", "trained"], help="Action generation mode")
    parser.add_argument("--save_frequency", type=int, default=1, help="Save every N steps")
    
    args = parser.parse_args()
    
    # Convert to dataclass
    collection_args = CollectionArgs(
        target_dir=args.target_dir,
        env_id=args.env_id,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        seed=args.seed,
        image_size=args.image_size,
        action_mode=args.action_mode,
        save_frequency=args.save_frequency
    )
    
    # Create collector and run
    collector = EnhancedDataCollector(collection_args)
    collector.collect_data()


if __name__ == "__main__":
    main() 
