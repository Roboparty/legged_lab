from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm
    
def reset_from_ref(
    env: ManagerBasedAnimationEnv, 
    env_ids: torch.Tensor, 
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    
    position = animation_term.get_root_pos_w(env_ids)[:, 0, :] + env.scene.env_origins[env_ids, :]
    orientation = animation_term.get_root_quat(env_ids)[:, 0, :]
    lin_vel = animation_term.get_root_vel_w(env_ids)[:, 0, :]
    ang_vel = animation_term.get_root_ang_vel_w(env_ids)[:, 0, :]
    
    pos = torch.cat([position, orientation], dim=-1)
    vel = torch.cat([lin_vel, ang_vel], dim=-1)
    
    robot.write_root_pose_to_sim(pos, env_ids=env_ids)
    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)
    

    
    
    
    