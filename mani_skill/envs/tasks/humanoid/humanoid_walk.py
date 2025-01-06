import copy
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBodyWithHeadCamera
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


@register_env("HumanoidWalk-v1", max_episode_steps=200)
class HumanoidWalkEnv(BaseEnv):
    """
    **Task Description:**
    A humanoid robot must learn how to walk.

    **Randomizations:**
    - The humanoid's initial position and configuration are randomized.

    **Success Conditions:**
    - The humanoid robot is able to walk consistently across episodes.
    """

    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_legs"]
    agent: UnitreeG1UpperBodyWithHeadCamera

    def __init__(self, *args, **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1UpperBodyWithHeadCamera.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.1, 0, 0.755]
        self.init_robot_qpos = UnitreeG1UpperBodyWithHeadCamera.keyframes["standing"].qpos.copy()
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
        pose = sapien_utils.look_at([1.0, 0.0, 1.6], [0, 0.0, 0.65])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 3)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 0.0, 1.6], [0, 0.0, 0.65])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=np.pi / 3
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.ground = ground.build_ground(self.scene, mipmap_levels=7)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.agent.robot.set_qpos(self.init_robot_qpos)
            self.agent.robot.set_pose(self.init_robot_pose)

    def evaluate(self):
        # Compute success based on stability and forward movement
        velocity = self.agent.robot.velocity[:, 0]  # x-direction velocity
        stability = self.agent.is_standing()
        forward_progress = velocity > 0.1  # Moving forward
        return {
            "success": stability & forward_progress,
            "stability": stability,
            "velocity": velocity,
        }

    def _get_obs_extra(self, info: Dict):
        return {
            "robot_pose": self.agent.robot.pose.raw_pose,
            "velocity": self.agent.robot.velocity,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Reward for stability and forward movement
        stability_reward = torch.tensor(1.0 if info["stability"] else 0.0, device=self.device)
        forward_reward = info["velocity"] * 10.0  # Scale for better learning signal
        reward = stability_reward + forward_reward
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10.0
