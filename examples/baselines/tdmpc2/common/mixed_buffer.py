import torch
import numpy as np
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
from common.buffer import Buffer
from common.synthetic_data_loader import SyntheticDataLoader


class MixedBuffer(Buffer):
	"""
	Mixed replay buffer that samples from both replay buffer (80%) and synthetic data (20%).
	Extends the original Buffer class to add synthetic data mixing.
	"""

	def __init__(self, cfg, synthetic_data_dir=None):
		super().__init__(cfg)
		self.synthetic_ratio = getattr(cfg, 'synthetic_ratio', 0.2)  # 20% synthetic by default
		self.synthetic_data_loader = None
		
		if synthetic_data_dir is not None:
			try:
				self.synthetic_data_loader = SyntheticDataLoader(
					synthetic_data_dir, cfg, device=self._device
				)
				print(f"Mixed buffer initialized with {self.synthetic_ratio:.1%} synthetic data")
			except Exception as e:
				print(f"Warning: Could not load synthetic data from {synthetic_data_dir}: {e}")
				print("Falling back to regular buffer only")
				self.synthetic_data_loader = None

	def sample(self):
		"""Sample a batch mixing replay buffer and synthetic data."""
		total_batch_size = self.cfg.batch_size
		
		# Determine how many samples from each source
		if self.synthetic_data_loader is not None and self._num_eps > 0:
			# Calculate split
			synthetic_batch_size = int(total_batch_size * self.synthetic_ratio)
			buffer_batch_size = total_batch_size - synthetic_batch_size
			
			samples = []
			
			# Sample from replay buffer
			if buffer_batch_size > 0:
				# Temporarily adjust batch size for buffer sampling
				original_batch_size = self._batch_size
				self._batch_size = buffer_batch_size * (self.cfg.horizon + 1)
				self._sampler = SliceSampler(
					num_slices=buffer_batch_size,
					end_key=None,
					traj_key='episode',
					truncated_key=None,
					strict_length=True,
				)
				
				buffer_td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
				buffer_sample = self._prepare_batch(buffer_td)
				samples.append(('buffer', buffer_sample))
				
				# Restore original settings
				self._batch_size = original_batch_size
				self._sampler = SliceSampler(
					num_slices=self.cfg.batch_size,
					end_key=None,
					traj_key='episode',
					truncated_key=None,
					strict_length=True,
				)
			
			# Sample from synthetic data
			if synthetic_batch_size > 0:
				synthetic_sample = self.synthetic_data_loader.sample_batch(synthetic_batch_size)
				if synthetic_sample is not None:
					samples.append(('synthetic', synthetic_sample))
			
			# Combine samples
			if len(samples) > 1:
				return self._combine_samples(samples)
			elif len(samples) == 1:
				return samples[0][1]
		
		# Fallback to regular buffer sampling
		if self._num_eps > 0:
			td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
			return self._prepare_batch(td)
		else:
			# Return dummy data if no data available
			return self._create_dummy_batch()

	def _combine_samples(self, samples):
		"""Combine samples from different sources into a single batch."""
		obs_list, action_list, reward_list, task_list = [], [], [], []
		
		for source, (obs, action, reward, task) in samples:
			obs_list.append(obs)
			action_list.append(action)
			reward_list.append(reward)
			if task is not None:
				task_list.append(task)
		
		# Concatenate along batch dimension
		if isinstance(obs_list[0], TensorDict):
			# Handle dict observations
			combined_obs = {}
			for key in obs_list[0].keys():
				combined_obs[key] = torch.cat([obs[key] for obs in obs_list], dim=1)
			combined_obs = TensorDict(combined_obs, batch_size=(), device=self._device)
		else:
			# Handle tensor observations
			combined_obs = torch.cat(obs_list, dim=1)
		
		combined_action = torch.cat(action_list, dim=1)
		combined_reward = torch.cat(reward_list, dim=1)
		combined_task = torch.cat(task_list, dim=1) if task_list else None
		
		return combined_obs, combined_action, combined_reward, combined_task

	def _create_dummy_batch(self):
		"""Create dummy batch when no data is available."""
		batch_size = self.cfg.batch_size
		horizon = self.cfg.horizon
		
		if self.cfg.obs == 'rgb':
			# Create dummy RGB observation
			obs = TensorDict({
				'rgb': torch.zeros((horizon + 1, batch_size, 3, 128, 128), device=self._device)
			}, batch_size=(), device=self._device)
		else:
			# Create dummy state observation
			obs = torch.zeros((horizon + 1, batch_size, 32), device=self._device)  # Adjust obs_dim as needed
		
		action = torch.zeros((horizon, batch_size, self.cfg.action_dim), device=self._device)
		reward = torch.zeros((horizon, batch_size, 1), device=self._device)
		task = None
		
		return obs, action, reward, task

	def get_synthetic_ratio(self):
		"""Return the current synthetic data ratio."""
		return self.synthetic_ratio if self.synthetic_data_loader is not None else 0.0 