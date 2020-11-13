import os
import time
from itertools import count
from typing import Sequence, Optional

import gym
import numpy as np
import pybullet
import pybullet_data

from pybullet_utils.bullet_client import BulletClient

from multi_goal.envs import Simulator, SettableGoalEnv, normalizer, SimObs
from multi_goal.envs.pybullet_imgs import get_labyrinth_cam_settings


class Labyrinth(SettableGoalEnv):
    def __init__(self, visualize=False, max_episode_len=100, *args, **kwargs):
        simulator = PyBullet(visualize=visualize)
        super().__init__(sim=simulator, max_episode_len=max_episode_len, *args, **kwargs)


class HardLabyrinth(SettableGoalEnv):
    def __init__(self, visualize=False, max_episode_len=100, *args, **kwargs):
        simulator = PyBullet(visualize=visualize, labyrinth=HardLabyrinthConfig())
        super().__init__(sim=simulator, max_episode_len=max_episode_len, *args, **kwargs)


BALL_RADIUS = 0.3


class SimpleLabyrinthConfig:
    fname = "assets/labyrinth.urdf"
    position = [7.5, -5/2, 1.5/2]
    wall_thickness = 0.31
    ball_radius = BALL_RADIUS
    lower_bound = np.array([-5/2 + wall_thickness/2, -5/2 + wall_thickness/2]) + ball_radius -0.01
    upper_bound = np.array([20 - 5/2 -wall_thickness/2, 10 - 5/2 - wall_thickness/2]) - ball_radius + 0.01
    agent_initial_pos = np.array([0, 0, ball_radius])
    goal_initial_pos = [2, 0, ball_radius]
    arrow_initial_pos = [*agent_initial_pos[:2], 2*ball_radius]
    getCamSettings = staticmethod(get_labyrinth_cam_settings)


class HardLabyrinthConfig(SimpleLabyrinthConfig):
    def __init__(self):
        self.fname = "assets/hard-labyrinth.urdf"
        wall_len = 16
        self.position = [0, 0, 0]
        margin = self.wall_thickness/2 + self.ball_radius - 0.01
        self.lower_bound = np.array([-wall_len, -wall_len])/2 + margin
        self.upper_bound = np.array([wall_len, wall_len])/2 - margin
        self.agent_initial_pos = np.array([-wall_len*3/8, -wall_len/8, self.ball_radius])
        self.goal_initial_pos = [-wall_len*3/8, wall_len/8, self.ball_radius]
        self.arrow_initial_pos = [*self.agent_initial_pos[:2], 2*self.ball_radius]


class PyBullet(Simulator):
    action_dim = 2
    _viz_lock_taken = False
    __filelocation__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    _red_ball_fname = os.path.join(__filelocation__, 'assets/little_ball.urdf')
    _green_ball_fname = os.path.join(__filelocation__, 'assets/immaterial_ball.urdf')
    _arrow_fname = os.path.join(__filelocation__, "assets/arrow.urdf")

    def __init__(self, visualize=False, labyrinth=SimpleLabyrinthConfig()):
        min_vel, max_vel = (-np.inf, np.inf)
        min_time, max_time = (-1, 1)
        self.observation_space = gym.spaces.Dict(spaces={
            "observation": gym.spaces.Box(low=np.array([min_vel, min_vel, min_time]), high=np.array([max_vel, max_vel, max_time])),
            "desired_goal": gym.spaces.Box(low=-1, high=1, shape=(2, )),
            "achieved_goal": gym.spaces.Box(low=-1, high=1, shape=(2, ))
        })
        self._visualize = visualize
        if visualize:
            assert not self._viz_lock_taken, "only one PyBullet simulation can be visualized simultaneously"
            PyBullet._viz_lock_taken = True

        self._bullet = BulletClient(connection_mode=pybullet.GUI if visualize else pybullet.DIRECT)
        self._bullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        self._bullet.setGravity(0, 0, -9.81)
        self._bullet.loadURDF("plane.urdf")
        self._agent_ball = self._bullet.loadURDF(self._red_ball_fname, labyrinth.agent_initial_pos)
        self._goal_ball = self._bullet.loadURDF(self._green_ball_fname, labyrinth.goal_initial_pos, useFixedBase=1)
        labyrinth_fname = os.path.join(self.__filelocation__, labyrinth.fname)
        self._bullet.loadURDF(labyrinth_fname, labyrinth.position, useFixedBase=1)
        self._arrow = self._bullet.loadURDF(self._arrow_fname, labyrinth.arrow_initial_pos, useFixedBase=1)

        self._ball_radius = labyrinth.ball_radius
        self._norm, self._denorm = normalizer(labyrinth.lower_bound, labyrinth.upper_bound)
        self.normed_starting_agent_obs = self._norm(labyrinth.agent_initial_pos[:2])
        self._cam_settings = labyrinth.getCamSettings(self._bullet)
        self._goal_img = None

    def step(self, action: np.ndarray) -> SimObs:
        sim_step_per_sec = 240
        agent_actions_per_sec = 10
        for _ in range((sim_step_per_sec // agent_actions_per_sec) - 1):
            if self._visualize:
                self._update_force_arrow_viz(force=action)
                time.sleep(1 / sim_step_per_sec)
            _apply_force(self._bullet, obj=self._agent_ball, force=action)
            self._bullet.stepSimulation()

        prev_agent_pos = self._get_agent_pos()
        _apply_force(self._bullet, obj=self._agent_ball, force=action)
        self._bullet.stepSimulation()
        agent_pos = self._get_agent_pos()
        agent_vel = 30*(agent_pos - prev_agent_pos)  # goal is to output a max vel of ~1
        rgb_img = self._get_cam_img()
        return SimObs(agent_pos=self._norm(agent_pos), obs=agent_vel, image=rgb_img)

    def _get_cam_img(self) -> np.ndarray:
        return self._bullet.getCameraImage(**self._cam_settings)[2]

    def _get_agent_pos(self) -> np.ndarray:
        return _position(self._bullet.getBasePositionAndOrientation(self._agent_ball))

    def _update_force_arrow_viz(self, force: np.ndarray) -> None:
        xforce, yforce = force
        yaw = np.angle(complex(xforce, yforce))
        quaternion = self._bullet.getQuaternionFromEuler([0, 0, yaw])
        agent_pos = self._get_agent_pos()
        _reset_object(self._bullet, self._arrow, [*agent_pos, 2*self._ball_radius], quaternion=quaternion)

    def set_agent_pos(self, pos: np.ndarray) -> None:
        pos = self._denorm(pos)
        _reset_object(self._bullet, self._agent_ball, pos=[*pos, self._ball_radius])

    def set_goal_pos(self, pos: np.ndarray) -> Optional[np.ndarray]:
        pos = self._denorm(pos)
        _reset_object(self._bullet, self._goal_ball, pos=[*pos, self._ball_radius])
        return self._mk_goal_img(goal_pos=pos)

    def _mk_goal_img(self, goal_pos) -> np.ndarray:
        agent_pos_backup = self._get_agent_pos()
        _reset_object(self._bullet, self._agent_ball, pos=[*goal_pos, self._ball_radius])
        img = self._get_cam_img()
        _reset_object(self._bullet, self._agent_ball, pos=[*agent_pos_backup, self._ball_radius])
        return img

    def is_success(self, achieved: np.ndarray, desired: np.ndarray) -> bool:
        achieved, desired = self._denorm(achieved), self._denorm(desired)
        return _goals_are_close(achieved_goal=achieved, desired_goal=desired)

    def render(self, *args, **kwargs):
        pass


def distance(x1: np.ndarray, x2: np.ndarray):
    return np.linalg.norm(x1 - x2) ** 2


def _goals_are_close(achieved_goal: np.ndarray, desired_goal: np.ndarray):
    ε = BALL_RADIUS
    return distance(achieved_goal, desired_goal) < ε


def _position(position_and_orientation: Sequence[float]) -> np.ndarray:
    return np.array(position_and_orientation[0])[:2]  # [pos, quaternion]


def _reset_object(bc: BulletClient, obj, pos: Sequence[float], quaternion=None):
    quaternion = quaternion if quaternion else [0, 0, 0, 1]
    bc.resetBasePositionAndOrientation(obj, pos, quaternion)


_zforce = 0
_force_multiplier = 5  # tuned value
def _apply_force(bc: BulletClient, obj, force: Sequence[float]):
    force = _force_multiplier * np.array([*force, _zforce])
    obj_pos, _  = bc.getBasePositionAndOrientation(obj)
    bc.applyExternalForce(objectUniqueId=obj, linkIndex=-1,
                          forceObj=force, posObj=obj_pos, flags=pybullet.WORLD_FRAME)


if __name__ == '__main__':
    env = Labyrinth(visualize=True)
    obs = env.reset()
    for t in count():
        action = obs.desired_goal - obs.achieved_goal
        obs = env.step(action / np.linalg.norm(action))[0]
        print(f"step {t}, vel: f{obs.observation}")
