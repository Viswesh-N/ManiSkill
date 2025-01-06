from typing import Any, Dict
import copy
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
        self.init_robot_qpos = None  # To be initialized with small noise
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
        """
        Dense reward for walking:
        - Reward forward progress.
        - Penalize instability or falling.
        """
        forward_velocity = info["velocity"] * 10.0  # Encourage forward movement
        stability_bonus = 1.0 if info["stability"] else -1.0  # Bonus for being upright
        reward = forward_velocity + stability_bonus

        # Penalize high joint torques (optional for smoother walking)
        if "joint_torques" in obs:
            torque_penalty = torch.sum(torch.abs(obs["joint_torques"]), dim=-1) * -0.01
            reward += torque_penalty

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Normalize dense reward for training stability.
        """
        return self.compute_dense_reward(obs, action, info) / 10.0
