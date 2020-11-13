import os
from abc import ABC
from typing import Type, Mapping

import math
import time
from itertools import repeat, chain, count
import matplotlib.pyplot as plt

import gym
import numpy as np
from pybullet_robots.panda.panda_sim_grasp import PandaSim
from pybullet_robots.panda.panda_sim_grasp import ll, ul, jr, rp, pandaEndEffectorIndex, pandaNumDofs
from pybullet_utils.bullet_client import BulletClient
import pybullet_data as pd
import pybullet

from multi_goal.envs import Simulator, SimObs, SettableGoalEnv, normalizer
from multi_goal.utils import get_updateable_scatter


class PandaEnv(SettableGoalEnv):
    def __init__(self, visualize=False, seed=0, max_episode_len=100, use_random_starting_pos=False):
        super().__init__(sim=PandaSimulator(visualize=visualize, task=PandaReachTask), max_episode_len=max_episode_len, seed=seed,
                         use_random_starting_pos=use_random_starting_pos)


class PandaPickAndPlace(SettableGoalEnv):
    def __init__(self, visualize=False, seed=0, max_episode_len=100, use_random_starting_pos=False):
        super().__init__(sim=PandaSimulator(visualize=visualize, task=PandaPickAndPlaceTask), max_episode_len=max_episode_len,
                         seed=seed, use_random_starting_pos=use_random_starting_pos)


_goal_space_bound = np.sqrt((0.8 ** 2) / 3)
_goal_low = np.array([-0.25, 0.01, -0.7])
_goal_high = np.array([0.25, _goal_space_bound, -_goal_space_bound])  # dont hit robot base
_normalize, _denormalize = normalizer(low=_goal_low, high=_goal_high)


class PandaTask(ABC):
    can_control_gripper: bool
    goal_is_endeffector: bool


class PandaReachTask(PandaTask):
    can_control_gripper = False
    goal_is_endeffector = True


class PandaPickAndPlaceTask(PandaTask):
    can_control_gripper = True
    goal_is_endeffector = False


class PandaSimulator(Simulator):
    _num_joints = 12
    _arm_joints = list(range(pandaNumDofs))
    _grasp_joints = [9, 10]
    __filelocation__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    _green_ball_fname = os.path.join(__filelocation__, "assets/immaterial_ball.urdf")
    _red_block_fname = os.path.join(__filelocation__, "assets/red_block.urdf")

    def __init__(self, task: Type[PandaTask], visualize=False):
        self._task = task
        self._all_joint_idxs = set(self._arm_joints + (self._grasp_joints if task.can_control_gripper else []))
        self._p = p = BulletClient(connection_mode=pybullet.GUI if visualize else pybullet.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, -9.8, 0)
        self._pandasim = PandaSim(bullet_client=p, offset=[0, 0, 0])
        load_orn = p.getQuaternionFromEuler([-np.pi / 2, np.pi / 2, 0])
        p.loadURDF("table/table.urdf", [0, -0.625, -0.5], baseOrientation=load_orn, useFixedBase=1)
        if not task.goal_is_endeffector:
            self._block_id = p.loadURDF(self._red_block_fname, [0, 0, -0.5], baseOrientation=load_orn, globalScaling=0.03)
        self._remove_unnecessary_objects()

        abs_lego_starting_pos = [0, 0.015, -0.5]
        abs_lego_starting_euler_orn = [-np.pi/2, 0, 0]
        p.resetBasePositionAndOrientation(self._pandasim.legos[0], abs_lego_starting_pos, p.getQuaternionFromEuler(abs_lego_starting_euler_orn))

        achieved_pos = self._get_endeffector_pos() if task.goal_is_endeffector else self._get_block_pos()
        self.normed_starting_agent_obs = _normalize(achieved_pos)

        self._original_joint_states = self._p.getJointStates(self._pandasim.panda, range(self._num_joints))
        self._pointing_down_orn = self._p.getQuaternionFromEuler([math.pi/2., 0., 0.])
        self._goal_pos = np.zeros(3)
        self._goal_ball_id = p.loadURDF(self._green_ball_fname, basePosition=self._goal_pos, useFixedBase=1, globalScaling=1 / 8)

        goal_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        min_vel, max_vel = (-np.inf, np.inf)
        min_time, max_time = (-1, 1)
        min_obs = [*repeat(min_vel, 2*len(self._all_joint_idxs)), min_time]  # 2 = pos + vel
        max_obs = [*repeat(max_vel, 2*len(self._all_joint_idxs)), max_time]
        obs_space = gym.spaces.Box(low=np.array(min_obs), high=np.array(max_obs)) if task.goal_is_endeffector else gym.spaces.Box(-1, 1, shape=(goal_space.shape[0] + 1, ))
        self.observation_space = gym.spaces.Dict(spaces={
            "observation": obs_space,
            "desired_goal": goal_space,
            "achieved_goal": goal_space
        })
        self.action_dim = 4 if task.can_control_gripper else 3  # [dx, dy, dz, dgrasp] as a change in gripper 3d-positon, where the y-axis references the height"""
        self._sim_steps_per_timestep = 10
        self._do_visualize = visualize
        self._plot = None

    def _remove_unnecessary_objects(self):
        spheres_ids = [4, 5, 6]
        legos_ids = [1, 2, 3]
        tray_id = [0]
        [self._p.removeBody(e) for e in spheres_ids + legos_ids + tray_id]

    def step(self, action: np.ndarray) -> SimObs:
        movement_factor = 1/25
        gripper_action = action[3] if self._task.can_control_gripper else 0
        action = np.append((np.array(action)[:3] * movement_factor), gripper_action)
        cur_pos, *_ = self._p.getLinkState(self._pandasim.panda, pandaEndEffectorIndex)
        pos = [coord+delta for coord, delta in zip(cur_pos, action)]

        grasp_forces = [50, 50]
        forces = chain(repeat(5*20, pandaNumDofs), grasp_forces if self._task.can_control_gripper else [])

        for idx in range(self._sim_steps_per_timestep):
            if idx % 5 == 0:
                desired_poses = self._p.calculateInverseKinematics(
                    self._pandasim.panda, pandaEndEffectorIndex, pos, self._pointing_down_orn, ll, ul, jr, rp, maxNumIterations=20)
                desired_poses = chain(desired_poses[:-2], [gripper_action, gripper_action] if self._task.can_control_gripper else [])

            for joint, pose, force in zip(self._all_joint_idxs, desired_poses, forces):
                self._p.setJointMotorControl2(self._pandasim.panda, joint, self._p.POSITION_CONTROL, pose, force=force)
            self._p.stepSimulation()
            if self._do_visualize:
                time.sleep(1/240)

        achieved_pos = self._get_endeffector_pos() if self._task.goal_is_endeffector else self._get_block_pos()
        obs = self._get_joints_info() if self._task.goal_is_endeffector else _normalize(self._get_endeffector_pos())
        return SimObs(agent_pos=_normalize(achieved_pos), obs=obs, image=np.empty(0))

    def _get_endeffector_pos(self):
        endeffector_pos = self._p.getLinkState(self._pandasim.panda, pandaEndEffectorIndex)[0]
        return np.array(endeffector_pos)

    def _get_block_pos(self) -> np.ndarray:
        pos, _ = self._p.getBasePositionAndOrientation(self._block_id)
        return np.array(pos)

    def _get_joints_info(self) -> np.ndarray:
        joint_states = self._p.getJointStates(self._pandasim.panda, range(self._num_joints))
        pos, vels, *_ = zip(*[s for idx, s in enumerate(joint_states) if idx in self._all_joint_idxs])
        return np.array([*pos, *vels])

    def set_agent_pos(self, pos: np.ndarray) -> None:
        # TODO: "set agent pos" does not apply to PandaPickAndPlace. More like "set achieved goal"
        pos = _denormalize([pos])
        self._reset_panda_to_original_joint_states()
        if self._task.goal_is_endeffector:
            self._reset_panda_to_pos(pos)
        else:
            self._p.resetBasePositionAndOrientation(self._block_id, pos, self._pointing_down_orn)

    def _reset_panda_to_pos(self, pos):
        joint_poses = self._p.calculateInverseKinematics(
            self._pandasim.panda, pandaEndEffectorIndex, pos, self._pointing_down_orn, ll, ul, jr, rp)
        for idx in range(pandaNumDofs):
            self._p.resetJointState(self._pandasim.panda, idx, joint_poses[idx])

    def set_goal_pos(self, pos: np.ndarray) -> None:
        self._goal_pos = _denormalize(pos)
        self._p.resetBasePositionAndOrientation(self._goal_ball_id, self._goal_pos, self._pointing_down_orn)

    _good_enough_distance = np.linalg.norm([0.03, 0.03, 0.03])
    def is_success(self, achieved: np.ndarray, desired: np.ndarray) -> bool:
        achieved, desired = _denormalize(achieved), _denormalize(desired)
        distance = np.linalg.norm(np.subtract(achieved, desired))
        return distance <= self._good_enough_distance

    def render(self, other_positions: Mapping[str, np.ndarray] = None,
               show_agent_and_goal_pos=False):
        if self._plot is None:
            fig, ax, scatter_fn = get_updateable_scatter(three_dim=True)
            ax.set_autoscale_on(True)
            xmin, zmin, ymin = _goal_low
            xmax, zmax, ymax = _goal_high
            ax.set_xlim3d(xmin, xmax)
            ax.set_ylim3d(ymin, ymax)
            ax.set_zlim3d(zmin, zmax)
            fig.show()
            self._plot = fig, ax, scatter_fn
        fig, ax, scatter_fn = self._plot

        if other_positions is not None:
            for color, positions in other_positions.items():
                scatter_fn(name=color, pts=None)  # clear previous
                if len(positions) > 0:
                    scatter_fn(name=color, pts=_rotate(_denormalize(positions)), c=color)

        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
        return fig, ax

    def _reset_panda_to_original_joint_states(self):
        joint_pos_and_vels = [(state[0], state[1]) for state in self._original_joint_states]
        for idx, (pos, vel) in enumerate(joint_pos_and_vels):
            self._p.resetJointState(self._pandasim.panda, idx, pos, vel)


def _rotate(pos: np.ndarray) -> np.ndarray:
    y_axis_up_permutation = [0, 2, 1]  # y-axis is up.
    if len(pos.shape) == 1:
        return pos[y_axis_up_permutation]
    return pos[:, y_axis_up_permutation]


def keyboard_control():
    """Returns an action based on key control.
    horizontal: {w,a,s,d} vertical: {y,x},  gripper: {c}"""
    cmd = input("press one key to move: {w,a,s,d,x,y,c}")
    red, green, blue, grip = 0, 0, 0, 0  # Axis colors
    if 'w' in cmd:
        red += 1
    if 's' in cmd:
        red -= 1
    if 'a' in cmd:
        blue -= 1
    if 'd' in cmd:
        blue += 1
    if 'y' in cmd:
        green -= 1
    if 'x' in cmd:
        green += 1
    if 'c' in cmd:
        grip += 1
    return np.array([red, green, blue, grip])


if __name__ == '__main__':
    env = PandaPickAndPlace(visualize=True, max_episode_len=10000)
    done = False
    while not done:
        obs = env.reset()
        for t in count():
            action = keyboard_control()
            obs, reward, done, info = env.step(action=action)
            print(done, obs.achieved_goal.round(2), obs.desired_goal.round(2), f"time: {t}")
            if done:
                break