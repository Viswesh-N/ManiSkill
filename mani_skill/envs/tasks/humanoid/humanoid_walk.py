from typing import Any, Dict
import sapien
import torch
import numpy as np

from mani_skill.agents.robots import UnitreeG1Simplified
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


@register_env("HumanoidWalk-v1", max_episode_steps=500)
class HumanoidWalkEnv(BaseEnv):
    """
    **Task Description:**
    Teach a Unitree G1 robot to walk forward stably over flat terrain.

    **Reward Modes:**
    - Dense: Encourages forward progress, stability, and penalizes falling or erratic movements.

    **Supported Robot:**
    - UnitreeG1Simplified
    """

    SUPPORTED_ROBOTS = ["unitree_g1_simplified_legs"]
    agent: UnitreeG1Simplified

    def __init__(self, *args, **kwargs):
        self.init_robot_pose = sapien.Pose(p=[0.0, 0.0, 0.755])  # Default standing pose
        self.contact_forces = None
        self.feet_pos = None
        self.feet_vel = None
        super().__init__(*args, robot_uids="unitree_g1_simplified_legs", **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
            scene_config=SceneConfig(contact_offset=0.02),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([2.0, 0.0, 1.5], [0.0, 0.0, 0.75])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 3)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([2.0, 0.0, 1.5], [0.0, 0.0, 0.75])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=np.pi / 3
        )

    def _load_scene(self, options: dict):
        build_ground(self.scene)  # Flat ground for walking

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """
        Initialize the robot with small random noise in its initial standing pose.
        """
        with torch.device(self.device):
            b = len(env_idx)
            standing_keyframe = self.agent.keyframes["standing"]
            random_qpos = (
                torch.randn(size=(b, self.agent.robot.dof[0]), dtype=torch.float) * 0.02
            )
            random_qpos += common.to_tensor(standing_keyframe.qpos, device=self.device)
            self.agent.robot.set_qpos(random_qpos)
            self.agent.robot.set_pose(self.init_robot_pose)

    def evaluate(self):
        """
        Evaluate the robot's performance.
        """
        velocity = self.agent.robot.get_root_linear_velocity()[:, 0]  # Forward velocity (x-axis)
        stability = self.agent.is_standing()
        progress = velocity > 0.2  # Encourage consistent forward progress
        return {
            "success": stability & progress,
            "stability": stability,
            "velocity": velocity,
        }

    def _get_obs_extra(self, info: Dict):
        """
        Additional observations for the agent to help policy learning.
        """
        return {
            "robot_pose": self.agent.robot.pose.raw_pose,
            "velocity": self.agent.robot.get_root_linear_velocity(),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        reward = torch.zeros(self.num_envs, device=self.device)

        # Stage 1: Height-Based Reward
        height = self.agent.robot.pose.p[:, 2]  # Torso height (z-axis)

        # Reward for increasing height 
        target_height = 0.9  # Target height for stable walking
        height_reward = torch.tanh(height - 0.5) * 5.0  # Encourage height above 0.5
        height_penalty = torch.where(
            height > 1.0, (height - 1.0) * -10.0,  # Penalize height too high
            torch.where(height < 0.5, (0.5 - height) * -10.0, 0.0)  # Penalize height too low
        )
        reward += height_reward + height_penalty

        # Bonus for being within the ideal height range
        upright_bonus = torch.where((height > 0.75) & (height < target_height), 2.0, 0.0)
        reward += upright_bonus

        # Stage 2: Forward Velocity Reward
        forward_velocity = self.agent.robot.get_root_linear_velocity()[:, 0]  # Forward velocity (x-axis)

        # Encourage forward velocity 
        target_velocity = 1.0  # Target forward velocity
        velocity_error = torch.abs(forward_velocity - target_velocity)
        velocity_reward = torch.exp(-velocity_error) * 10.0  # Reward approaches max as error approaches 0
        reward += velocity_reward

        # Stage 3: Penalize Instability
        torso_instability_penalty = torch.abs(height - target_height) * -5.0  # Penalize deviations from target height
        reward += torso_instability_penalty

        # Penalize jerky movements or abrupt changes
        torso_velocity = torch.norm(self.agent.robot.get_root_linear_velocity(), dim=1)
        smooth_motion_penalty = -torch.clamp(torso_velocity - 1.0, min=0.0) * 5.0
        reward += smooth_motion_penalty

        # Stage 4: Penalize Lateral Instability
        lateral_velocity = torch.abs(self.agent.robot.get_root_linear_velocity()[:, 1])  # Side-to-side motion (y-axis)
        lateral_penalty = -torch.clamp(lateral_velocity, 0.0, 0.5) * 5.0
        reward += lateral_penalty

        # Stage 5: Penalize Angular Instability (optional, based on rotation)
        torso_angular_velocity = torch.norm(self.agent.robot.get_root_angular_velocity(), dim=1)
        angular_instability_penalty = -torch.clamp(torso_angular_velocity, 0.0, 1.0) * 5.0
        reward += angular_instability_penalty

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Normalize dense reward for training stability.
        """
        max_reward = 25.0
        return self.compute_dense_reward(obs, action, info) / max_reward
