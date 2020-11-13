import time
from typing import Mapping

import gym
import numpy as np
import matplotlib.pyplot as plt

from multi_goal.envs import normalizer, SettableGoalEnv, Simulator, SimObs
from multi_goal.utils import get_updateable_scatter, get_updateable_contour


class ToyLab(SettableGoalEnv):
    def __init__(self, max_episode_len=80, *args, **kwargs):
        super().__init__(sim=ToyLabSimulator(), max_episode_len=max_episode_len, *args, **kwargs)


class ToyLabSimulator(Simulator):
    action_dim = 2
    normed_starting_agent_obs = np.array([-.75, -.5])
    middle_wall_len = 12
    sidewall_height = 8
    labyrinth_corners = np.array([
        (0, 0),
        (-middle_wall_len, 0),
        (-middle_wall_len, -sidewall_height/2),
        (sidewall_height/2, -sidewall_height/2),
        (sidewall_height/2, sidewall_height/2),
        (-middle_wall_len, sidewall_height/2),
        (-middle_wall_len, 0)
    ])

    _labyrinth_lower_bound = np.array([-middle_wall_len, -sidewall_height / 2])
    _labyrinth_upper_bound = np.array([sidewall_height / 2, sidewall_height / 2])
    _normalize, _denormalize = normalizer(_labyrinth_lower_bound, _labyrinth_upper_bound)
    _normalize, _denormalize = staticmethod(_normalize), staticmethod(_denormalize)

    def __init__(self):
        self.observation_space = gym.spaces.Dict(spaces={
            "observation": gym.spaces.Box(low=-1, high=1, shape=(1, )),  # time feature
            "desired_goal": gym.spaces.Box(low=-1, high=1, shape=(2, )),
            "achieved_goal": gym.spaces.Box(low=-1, high=1, shape=(2, ))
        })
        # TODO: agent_pos and goal_pos are not part of the public API.
        self.agent_pos = None
        self.goal_pos = None
        self._plot = None

    def step(self, action: np.ndarray) -> SimObs:
        self.agent_pos = self._simulation_step(self.agent_pos, action)
        return SimObs(agent_pos=self._normalize(self.agent_pos), obs=np.empty(0), image=np.empty(0))

    _step_len = 0.5
    def _simulation_step(self, cur_pos: np.ndarray, action: np.ndarray) -> np.ndarray:
        assert cur_pos.shape == action.shape
        small_val = 1e-6
        x1, x2 = cur_pos + self._step_len * action

        # no pass through 0
        if all(cur_pos <= 0):
            x2 = min(-small_val, x2)
        if cur_pos[0] <= 0 and cur_pos[1] >= 0:
            x2 = max(small_val, x2)

        return np.clip(np.array([x1, x2]), a_min=self._labyrinth_lower_bound, a_max=self._labyrinth_upper_bound)

    def set_agent_pos(self, pos: np.ndarray) -> None:
        self.agent_pos = self._denormalize(pos)

    def set_goal_pos(self, pos: np.ndarray) -> None:
        self.goal_pos = self._denormalize(pos)

    def is_success(self, achieved: np.ndarray, desired: np.ndarray) -> bool:
        return self._is_success(achieved_pos=self._denormalize(achieved),
                                desired_pos=self._denormalize(desired))

    def _is_success(self, achieved_pos: np.ndarray, desired_pos: np.ndarray) -> bool:
        return (self._are_on_same_side_of_wall(achieved_pos, desired_pos) and
                self._are_close(achieved_pos, desired_pos))

    @classmethod
    def _are_on_same_side_of_wall(cls, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        return cls._is_above_wall(pos1) == cls._is_above_wall(pos2)

    @staticmethod
    def _is_above_wall(pos: np.ndarray) -> bool:
        return pos[1] > 0

    _max_single_action_dist = np.linalg.norm(np.ones(action_dim)) * _step_len
    def _are_close(self, x1: np.ndarray, x2: np.ndarray) -> bool:
        return np.linalg.norm(x1 - x2)**2 < self._max_single_action_dist

    def render(self, other_positions: Mapping[str, np.ndarray] = None,
               show_agent_and_goal_pos=True, positions_density=None):
        if self._plot is None:
            self._plot = fig, ax, scatter_fn = get_updateable_scatter()
            ax.plot(*self.labyrinth_corners.T)
            fig.show()

        fig, ax, scatter_fn = self._plot

        if positions_density is not None:
            plot_contour_fn(unnormed_data=self._denormalize(positions_density), ax=ax)

        if other_positions is not None:
            for color, positions in other_positions.items():
                scatter_fn(name=color, pts=None)  # clear previous
                if len(positions) > 0:
                    scatter_fn(name=color, pts=self._denormalize(positions), c=color)

        agent_pos = None if not show_agent_and_goal_pos else self.agent_pos
        goal_pos = None if not show_agent_and_goal_pos else self.goal_pos
        scatter_fn(name="agent_pos", pts=agent_pos)
        scatter_fn(name="goal", pts=goal_pos, c="green")

        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        return fig, ax


plot_contour_fn = get_updateable_contour()

if __name__ == '__main__':
    env = ToyLab(seed=1)
    env.reset()
    env.render()
    for _ in range(100):
        time.sleep(0.2)
        action = env.action_space.sample()
        env.step(action)
        env.render()
