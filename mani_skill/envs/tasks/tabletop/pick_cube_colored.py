from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, WidowXAI, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# Import the base PickCube environment
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv, PICK_CUBE_DOC_STRING


class ColoredTableSceneBuilder(TableSceneBuilder):
    """Table scene builder that allows custom table colors"""
    
    def __init__(self, env, robot_init_qpos_noise=0.02, table_color=[1, 1, 1, 1]):
        super().__init__(env, robot_init_qpos_noise)
        self.table_color = np.array(table_color)
    
    def build(self):
        super().build()
        # Change table color by modifying the render material
        for part in self.table._objs:
            for triangle in (
                part.find_component_by_type(sapien.render.RenderBodyComponent)
                .render_shapes[0]
                .parts
            ):
                triangle.material.set_base_color(self.table_color)
                # Optional: remove textures for solid color
                triangle.material.set_base_color_texture(None)
                triangle.material.set_normal_texture(None)
                triangle.material.set_emission_texture(None)
                triangle.material.set_transmission_texture(None)
                triangle.material.set_metallic_texture(None)
                triangle.material.set_roughness_texture(None)


class BlueTableSceneBuilder(ColoredTableSceneBuilder):
    def __init__(self, env, robot_init_qpos_noise=0.02):
        super().__init__(env, robot_init_qpos_noise, table_color=[0.2, 0.4, 0.8, 1.0])  # Blue


class YellowTableSceneBuilder(ColoredTableSceneBuilder):
    def __init__(self, env, robot_init_qpos_noise=0.02):
        super().__init__(env, robot_init_qpos_noise, table_color=[0.9, 0.9, 0.2, 1.0])  # Yellow


class GreenTableSceneBuilder(ColoredTableSceneBuilder):
    def __init__(self, env, robot_init_qpos_noise=0.02):
        super().__init__(env, robot_init_qpos_noise, table_color=[0.2, 0.8, 0.3, 1.0])  # Green


class RedTableSceneBuilder(ColoredTableSceneBuilder):
    def __init__(self, env, robot_init_qpos_noise=0.02):
        super().__init__(env, robot_init_qpos_noise, table_color=[0.8, 0.2, 0.2, 1.0])  # Red


class BlackTableSceneBuilder(ColoredTableSceneBuilder):
    def __init__(self, env, robot_init_qpos_noise=0.02):
        super().__init__(env, robot_init_qpos_noise, table_color=[0.1, 0.1, 0.1, 1.0])  # Black


class GrayTableSceneBuilder(ColoredTableSceneBuilder):
    def __init__(self, env, robot_init_qpos_noise=0.02):
        super().__init__(env, robot_init_qpos_noise, table_color=[0.5, 0.5, 0.5, 1.0])  # Gray


@register_env("PickCubeBlueTable-v1", max_episode_steps=50)
class PickCubeBlueTableEnv(PickCubeEnv):
    """PickCube environment with blue table"""
    
    def _load_scene(self, options: dict):
        self.table_scene = BlueTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)


@register_env("PickCubeYellowTable-v1", max_episode_steps=50)
class PickCubeYellowTableEnv(PickCubeEnv):
    """PickCube environment with yellow table"""
    
    def _load_scene(self, options: dict):
        self.table_scene = YellowTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)


@register_env("PickCubeGreenTable-v1", max_episode_steps=50)
class PickCubeGreenTableEnv(PickCubeEnv):
    """PickCube environment with green table"""
    
    def _load_scene(self, options: dict):
        self.table_scene = GreenTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)


@register_env("PickCubeRedTable-v1", max_episode_steps=50)
class PickCubeRedTableEnv(PickCubeEnv):
    """PickCube environment with red table"""
    
    def _load_scene(self, options: dict):
        self.table_scene = RedTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)


@register_env("PickCubeBlackTable-v1", max_episode_steps=50)
class PickCubeBlackTableEnv(PickCubeEnv):
    """PickCube environment with black table"""
    
    def _load_scene(self, options: dict):
        self.table_scene = BlackTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)


@register_env("PickCubeGrayTable-v1", max_episode_steps=50)
class PickCubeGrayTableEnv(PickCubeEnv):
    """PickCube environment with gray table"""
    
    def _load_scene(self, options: dict):
        self.table_scene = GrayTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)


PickCubeBlueTableEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="Panda") + "\n\n**Visual Variant:** Blue table surface."
PickCubeYellowTableEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="Panda") + "\n\n**Visual Variant:** Yellow table surface."
PickCubeGreenTableEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="Panda") + "\n\n**Visual Variant:** Green table surface."
PickCubeRedTableEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="Panda") + "\n\n**Visual Variant:** Red table surface."
PickCubeBlackTableEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="Panda") + "\n\n**Visual Variant:** Black table surface."
PickCubeGrayTableEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="Panda") + "\n\n**Visual Variant:** Gray table surface." 