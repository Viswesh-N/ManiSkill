import copy

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


@register_agent()
class SO101(BaseAgent):
    uid = "so101"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/so101/so101.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.5, dynamic_friction=2.5, restitution=0.0)
        ),
        link=dict(
            gripper_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            moving_jaw_so101_v1_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, -1.5708, 1.5708, 0.66, 0, -1.1]),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 6),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
        extended=Keyframe(
            qpos=np.array([0, -0.7854, 0.7854, 0, 0, -0.5]),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
    )

    arm_joint_names = [
        "shoulder_pan",
        "shoulder_lift", 
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    gripper_joint_names = [
        "gripper",
    ]

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=[1.2e3] * 6,  # Slightly higher stiffness for SO101
            damping=[1.2e2] * 6,
            force_limit=120,  # Higher force limit for SO101
            normalize_action=False,
        )

        # Improved delta position control for SO101
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.06, -0.06, -0.06, -0.06, -0.06, -0.25],  # Slightly larger delta for SO101
            [0.06, 0.06, 0.06, 0.06, 0.06, 0.25],
            stiffness=[1.2e3] * 6,
            damping=[1.2e2] * 6,
            force_limit=120,
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["gripper_link"]
        self.finger2_link = self.robot.links_map["moving_jaw_so101_v1_link"]
        self.finger1_tip = self.finger1_link
        self.finger2_tip = self.finger2_link

    @property
    def tcp_pos(self):
        # computes the tool center point as the mid point between the the fixed and moving jaw's tips
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_grasping(self, object: Actor, min_force=0.6, max_angle=105):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.6.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 105.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.15):
        """Check if the robot is static (improved for SO101)"""
        qvel = self.robot.get_qvel()[:, :-1]  # exclude the gripper joint
        return torch.max(torch.abs(qvel), 1)[0] <= threshold 