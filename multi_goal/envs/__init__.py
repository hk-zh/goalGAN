from abc import ABC
from collections import OrderedDict
from itertools import cycle
from typing import Tuple, Optional, Mapping, List, Sequence, NamedTuple

import gym
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typeguard import typechecked

GoalHashable = Tuple[float]


class Observation(OrderedDict):
    def __init__(self, observation: np.ndarray, achieved_goal: np.ndarray,
                 desired_goal: np.ndarray) -> None:
        super().__init__(observation=observation, achieved_goal=achieved_goal,
                         desired_goal=desired_goal)
        self.observation = observation
        self.achieved_goal = achieved_goal
        self.desired_goal = desired_goal
        self._tol = 1e-4

    def __eq__(self, other):
        return all(np.allclose(other[k], v, atol=self._tol) for k, v in self.items())

    def __ne__(self, other):
        return not self.__eq__(other)

    def _cmp(self, x, y):
        return np.allclose(x, y, atol=self._tol)

    def equal_except_time(self, other):
        return (self._cmp(self.observation[:-1], other.observation[:-1]) and
                self._cmp(self.achieved_goal, other.achieved_goal) and
                self._cmp(self.desired_goal, other.desired_goal))


class ISettableGoalEnv(ABC, gym.GoalEnv):
    max_episode_len: int
    starting_agent_pos: np.ndarray

    def set_possible_goals(self, goals: Optional[np.ndarray], entire_space=False) -> None:
        raise NotImplementedError

    def get_successes_of_goals(self) -> Mapping[GoalHashable, List[bool]]:
        raise NotImplementedError


def normalizer(low, high):
    dim = len(low)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit([low, high])

    def normalize(goal: Sequence[float]) -> np.ndarray:
        return scaler.transform(goal[np.newaxis])[0]

    def denormalize(norm_goals: Sequence[Sequence[float]]) -> np.ndarray:
        if not isinstance(norm_goals, np.ndarray):
            norm_goals = np.array(list(norm_goals))

        is_single_goal = norm_goals.size == dim
        if is_single_goal:
            norm_goals = norm_goals.reshape((1, dim))

        res = scaler.inverse_transform(norm_goals)
        if is_single_goal:
            res = res[0]

        return res

    return normalize, denormalize


SimObs = NamedTuple("State", fields=[("agent_pos", np.ndarray),
                                     ("obs", np.ndarray),
                                     ("image", Optional[np.ndarray])])


class Simulator:
    observation_space: gym.spaces.Dict
    action_dim: int
    normed_starting_agent_obs: np.ndarray

    def step(self, action: np.ndarray) -> SimObs:
        raise NotImplementedError

    def set_agent_pos(self, pos: np.ndarray) -> None:
        raise NotImplementedError

    def set_goal_pos(self, pos: np.ndarray) -> None:
        raise NotImplementedError

    def is_success(self, achieved: np.ndarray, desired: np.ndarray) -> bool:
        raise NotImplementedError

    def render(self, *args, **kwargs):
        raise NotImplementedError


class SettableGoalEnv(ISettableGoalEnv):
    reward_range = (-1, 0)

    def __init__(self, sim: Simulator, max_episode_len: int,
                 seed=0, use_random_starting_pos=False, obs_as_img=False):
        super().__init__()
        self._obs_as_img = obs_as_img
        self.observation_space = sim.observation_space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(sim.action_dim,))
        self.seed(seed)
        self.starting_agent_pos = sim.normed_starting_agent_obs
        self.max_episode_len = max_episode_len
        self._sim = sim
        self._possible_goals = None
        self._successes_per_goal: Mapping[GoalHashable, List[bool]] = dict()
        self._use_random_starting_pos = use_random_starting_pos
        self._agent_pos = None
        self._achieved_img = None
        self._goal_pos = None
        self._desired_img = None
        self._cur_obs = None
        self._step_num = 0
        self.reset()

    def _new_initial_pos(self) -> np.ndarray:
        if not self._use_random_starting_pos:
            return self.starting_agent_pos
        return self.observation_space["desired_goal"].sample()

    def _new_goal(self) -> np.ndarray:
        if self._possible_goals is None:
            return self.observation_space["desired_goal"].sample()
        return next(self._possible_goals)

    def step(self, action: np.ndarray):
        action = np.array(action)
        assert self.action_space.contains(action*0.99), f"Action is not within 1% bounds: {action}"
        self._step_num += 1
        self._agent_pos, self._cur_obs, self._achieved_img = self._sim.step(action=action)
        obs = self._make_obs_object()
        reward = self.compute_reward(obs.achieved_goal, obs.desired_goal, {})
        is_success = reward == max(self.reward_range)
        done = (is_success or self._step_num >= self.max_episode_len)
        if done and len(self._successes_per_goal) > 0:
            self._successes_per_goal[tuple(self._goal_pos)].append(is_success)
        return obs, reward, done, {"is_success": float(is_success)}

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """
        This function can take inputs from outside, so the inputs are the normalized
        versions of the goals (to [-1, 1]).
        """
        is_success = self._sim.is_success(achieved=achieved_goal, desired=desired_goal)
        return max(self.reward_range) if is_success else min(self.reward_range)

    def reset(self, reset_agent_pos=True) -> Observation:
        self._step_num = 0

        if reset_agent_pos:
            self._agent_pos = self._new_initial_pos()
            self._sim.set_agent_pos(self._agent_pos)

        self._goal_pos = self._new_goal()
        self._desired_img = self._sim.set_goal_pos(self._goal_pos)

        null_action = np.zeros(shape=self.action_space.shape)
        self._agent_pos, self._cur_obs, img = self._sim.step(action=null_action)
        return self._make_obs_object()

    def _make_obs_object(self) -> Observation:
        time_feature = (-2/self.max_episode_len)*self._step_num + 1
        if self._obs_as_img:
            return Observation(observation=np.array([time_feature]),
                               achieved_goal=self._achieved_img,
                               desired_goal=self._desired_img)
        return Observation(observation=np.append(self._cur_obs, time_feature),
                           achieved_goal=self._agent_pos,
                           desired_goal=self._goal_pos)

    @typechecked
    def set_possible_goals(self, goals: Optional[np.ndarray], entire_space=False) -> None:
        if goals is None and entire_space:
            self._possible_goals = None
            self._successes_per_goal = dict()
            return

        assert len(goals.shape) == 2, f"Goals must have shape (N, 2), instead: {goals.shape}"
        assert goals.shape[1] == self.observation_space["desired_goal"].shape[0]
        self._possible_goals = cycle(np.random.permutation(goals))
        self._successes_per_goal = {tuple(g): [] for g in goals}

    def get_successes_of_goals(self) -> Mapping[GoalHashable, List[bool]]:
        return dict(self._successes_per_goal)

    def seed(self, seed=None):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def render(self, *args, **kwargs):
        return self._sim.render(*args, **kwargs)


def dim_goal(env: gym.GoalEnv):
    return env.observation_space["desired_goal"].shape[0]