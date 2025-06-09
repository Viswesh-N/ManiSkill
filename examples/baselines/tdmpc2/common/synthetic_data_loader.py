import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tensordict.tensordict import TensorDict
import copy
import torchvision.transforms as transforms


class SyntheticDataLoader:
    """
    Loads synthetic data created by postprocess.py for mixed training with replay buffer.
    
    Data structure expected:
    synthetic_data_dir/
    ├── metadata/
    │   └── processed_metadata.json
    ├── images/
    │   └── *.png (edited images)
    └── masked_images/
        └── *.png (intermediate masked images)
    """
    
    def __init__(self, synthetic_data_dir, cfg, device='cuda'):
        self.synthetic_data_dir = Path(synthetic_data_dir)
        self.cfg = cfg
        self.device = device
        
        metadata_path = self.synthetic_data_dir / "metadata" / "processed_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Synthetic data metadata not found: {metadata_path}")
            
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        self.num_samples = len(self.metadata["edited_images"])
        print(f"Loaded synthetic dataset with {self.num_samples} samples")
        
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        """Convert synthetic data into TDMPC2-compatible sequences"""
        sequences = []
        
        episode_length = self.cfg.episode_length if hasattr(self.cfg, 'episode_length') else 50
        
        # Get expected image size from config
        expected_size = getattr(self.cfg, 'render_size', 64)
        
        # Create resize transform if needed
        resize_transform = transforms.Resize((expected_size, expected_size), antialias=True)
        
        current_seq = []
        for i in range(self.num_samples):
            # Load image
            img_path = self.synthetic_data_dir / self.metadata["edited_images"][i]
            image = np.array(Image.open(img_path))
            
            # Ensure image has 3 dimensions (H, W, C)
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]  # Remove alpha channel
            
            # Convert to tensor for resizing
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)  # [C, H, W]
            
            # Resize if needed
            if image_tensor.shape[-1] != expected_size or image_tensor.shape[-2] != expected_size:
                image_tensor = resize_transform(image_tensor)
            
            image_tensor = image_tensor / 255.0
            
            action = np.array(self.metadata["actions"][i])
            reward = self.metadata["rewards"][i] if "rewards" in self.metadata else 0.0
            terminated = self.metadata["terminations"][i] if "terminations" in self.metadata else False
            truncated = self.metadata["truncations"][i] if "truncations" in self.metadata else False
            
            # Create observation dict matching TDMPC2 format
            if self.cfg.obs == 'rgb':
                # Auto-detect expected channels from config if available
                expected_channels = None
                if hasattr(self.cfg, 'obs_shape') and 'rgb' in self.cfg.obs_shape:
                    expected_channels = self.cfg.obs_shape['rgb'][0]  # First dimension is channels
                
                if expected_channels is not None and expected_channels > 3:
                    # Frame stacking is expected - repeat the frame to match expected channels
                    num_repeats = expected_channels // 3
                    rgb_stacked = image_tensor.repeat(num_repeats, 1, 1)  # [expected_channels, H, W]
                    obs = {'rgb': rgb_stacked}
                else:
                    # No frame stacking or single frame expected
                    obs = {'rgb': image_tensor}
                
                # Add rgb-state if include_state is True
                if getattr(self.cfg, 'include_state', False):
                    # Generate dummy state data that matches expected dimensions
                    if hasattr(self.cfg, 'obs_shape') and 'rgb-state' in self.cfg.obs_shape:
                        state_shape = self.cfg.obs_shape['rgb-state']
                        # Create dummy state data (zeros for now) on the same device as image_tensor
                        dummy_state = torch.zeros(state_shape, dtype=torch.float32, device=image_tensor.device)
                        obs['rgb-state'] = dummy_state
                    else:
                        # Fallback: create a reasonable dummy state size (common ManiSkill state sizes)
                        # Default to a typical state size with frame stacking
                        state_dim = 32  # Typical robot state dimension
                        num_frames = expected_channels // 3 if expected_channels and expected_channels > 3 else 3
                        dummy_state = torch.zeros(state_dim * num_frames, dtype=torch.float32, device=image_tensor.device)
                        obs['rgb-state'] = dummy_state
            else:
                # Handle tensor observations (non-dict)
                # Auto-detect expected channels from config if available
                expected_channels = None
                if hasattr(self.cfg, 'obs_shape'):
                    obs_key = self.cfg.get('obs', 'state')
                    if obs_key in self.cfg.obs_shape:
                        expected_channels = self.cfg.obs_shape[obs_key][0]  # First dimension is channels
                
                if expected_channels is not None and expected_channels > 3:
                    # Frame stacking is expected - repeat the frame to match expected channels
                    num_repeats = expected_channels // 3
                    obs = image_tensor.repeat(num_repeats, 1, 1)  # [expected_channels, H, W]
                else:
                    # No frame stacking or single frame expected
                    obs = image_tensor
            
            step_data = {
                'obs': obs,
                'action': torch.from_numpy(action).float(),
                'reward': torch.tensor(reward).float(),
                'terminated': torch.tensor(terminated).bool(),
                'truncated': torch.tensor(truncated).bool()
            }
            
            current_seq.append(step_data)
            
            # End episode when we hit termination/truncation or reach max length
            if terminated or truncated or len(current_seq) >= episode_length:
                if len(current_seq) > 1:  # Only add sequences with at least 2 steps
                    sequences.append(current_seq)
                current_seq = []
        
        # Add remaining sequence if it exists
        if len(current_seq) > 1:
            sequences.append(current_seq)
        
        self.sequences = sequences
        print(f"Created {len(self.sequences)} synthetic episodes")
        
        # Debug: Print sample observation shape
        if len(self.sequences) > 0 and len(self.sequences[0]) > 0:
            sample_obs = self.sequences[0][0]['obs']
            if isinstance(sample_obs, dict):
                print("Synthetic data observation keys and shapes:")
                for key, value in sample_obs.items():
                    print(f"  {key}: {value.shape}")
            else:
                print(f"Synthetic data - obs shape: {sample_obs.shape}")
            print(f"Expected render size: {expected_size}x{expected_size}")
            print(f"Include state: {getattr(self.cfg, 'include_state', False)}")
    
    def sample_batch(self, batch_size):
        """Sample a batch of sequences for training"""
        if len(self.sequences) == 0:
            return None
            
        # Randomly sample episodes
        episode_indices = np.random.choice(len(self.sequences), size=batch_size, replace=True)
        
        batch_sequences = []
        for ep_idx in episode_indices:
            episode = self.sequences[ep_idx]
            
            # Sample a random subsequence of length horizon+1
            if len(episode) <= self.cfg.horizon + 1:
                # Episode is too short, pad or skip
                if len(episode) >= 2:
                    # Pad with the last observation
                    padded_episode = episode[:]
                    while len(padded_episode) <= self.cfg.horizon + 1:
                        # Create a proper deep copy of the last step
                        last_step = copy.deepcopy(episode[-1])
                        # Modify the copied tensors (create new tensors to avoid reference issues)
                        last_step['reward'] = torch.tensor(0.0).float()
                        last_step['terminated'] = torch.tensor(False).bool()
                        last_step['truncated'] = torch.tensor(False).bool()
                        padded_episode.append(last_step)
                    batch_sequences.append(padded_episode[:self.cfg.horizon + 1])
            else:
                # Sample random subsequence
                start_idx = np.random.randint(0, len(episode) - self.cfg.horizon)
                subseq = episode[start_idx:start_idx + self.cfg.horizon + 1]
                batch_sequences.append(subseq)
        
        if not batch_sequences:
            return None
            
        # Convert to TDMPC2 format
        return self._convert_to_tdmpc2_format(batch_sequences)
    
    def _convert_to_tdmpc2_format(self, sequences):
        """Convert sequences to TDMPC2's expected format to match regular buffer output"""
        batch_size = len(sequences)
        horizon = self.cfg.horizon
        
        # Stack observations - need all timesteps (horizon+1 total)
        if self.cfg.obs == 'rgb':
            # Handle dict observations
            obs_batch = {}
            for key in sequences[0][0]['obs'].keys():
                # Collect observations for all timesteps
                key_tensors = []
                for t in range(horizon + 1):
                    timestep_tensors = []
                    for seq in sequences:
                        if t < len(seq):
                            timestep_tensors.append(seq[t]['obs'][key])
                        else:
                            # Pad with last observation if sequence is shorter
                            timestep_tensors.append(seq[-1]['obs'][key])
                    key_tensors.append(torch.stack(timestep_tensors))  # [batch_size, ...]
                
                # Stack to get [horizon+1, batch_size, ...]
                obs_batch[key] = torch.stack(key_tensors).to(self.device)
            
            # Check if we should return a TensorDict or extract the main observation
            # If there's only one key (just 'rgb'), extract it as tensor to match buffer behavior
            if len(obs_batch.keys()) == 1 and 'rgb' in obs_batch:
                obs = obs_batch['rgb']  # Return tensor directly
            else:
                # Create TensorDict when multiple keys exist (e.g., rgb + rgb-state)
                obs = TensorDict(obs_batch, batch_size=(horizon+1, batch_size), device=self.device)
        else:
            # Handle tensor observations
            obs_tensors = []
            for t in range(horizon + 1):
                timestep_tensors = []
                for seq in sequences:
                    if t < len(seq):
                        timestep_tensors.append(seq[t]['obs'])
                    else:
                        # Pad with last observation if sequence is shorter
                        timestep_tensors.append(seq[-1]['obs'])
                obs_tensors.append(torch.stack(timestep_tensors))  # [batch_size, ...]
            
            # Stack to get [horizon+1, batch_size, ...]
            obs = torch.stack(obs_tensors).to(self.device)
        
        # Stack actions - need horizon timesteps (skip first observation)
        action_tensors = []
        for t in range(horizon):
            timestep_actions = []
            for seq in sequences:
                # Actions start from timestep 1 (after first observation)
                action_idx = t + 1
                if action_idx < len(seq):
                    timestep_actions.append(seq[action_idx]['action'])
                else:
                    # Pad with last action if sequence is shorter
                    timestep_actions.append(seq[-1]['action'])
            action_tensors.append(torch.stack(timestep_actions))  # [batch_size, action_dim]
        
        # Stack to get [horizon, batch_size, action_dim]
        action = torch.stack(action_tensors).to(self.device)
        
        # Stack rewards - need horizon timesteps (skip first observation)
        reward_tensors = []
        for t in range(horizon):
            timestep_rewards = []
            for seq in sequences:
                # Rewards start from timestep 1 (after first observation)
                reward_idx = t + 1
                if reward_idx < len(seq):
                    timestep_rewards.append(seq[reward_idx]['reward'])
                else:
                    # Pad with zero reward if sequence is shorter
                    timestep_rewards.append(torch.tensor(0.0).float())
            reward_tensors.append(torch.stack(timestep_rewards))  # [batch_size]
        
        # Stack to get [horizon, batch_size, 1]
        reward = torch.stack(reward_tensors).unsqueeze(-1).to(self.device)
        
        # Return task=None to match expected format (obs, action, reward, task)
        return obs, action, reward, None 