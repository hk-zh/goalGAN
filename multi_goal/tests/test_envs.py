from itertools import combinations

import gym
import pytest

from multi_goal.GenerativeGoalLearning import trajectory, null_agent
from multi_goal.envs import Observation
from multi_goal.envs.pybullet_labyrinth_env import Labyrinth, HardLabyrinth
from multi_goal.envs.pybullet_panda_robot import PandaEnv, PandaPickAndPlace
from multi_goal.envs.toy_labyrinth_env import normalizer, ToyLabSimulator, ToyLab
import numpy as np


def test_normalizer():
    rand = np.random.randn(2, 10)
    low, high = rand.max(axis=1), rand.min(axis=1)
    norm, denorm = normalizer(low, high)
    for _ in range(10):
        goal = np.random.uniform(low, high)
        assert np.allclose(norm(denorm(goal)), goal)


def test_are_on_same_side_of_wall():
    below_wall = np.array([1, -1])
    above_wall = np.array([1, 1])
    c = ToyLabSimulator
    assert c._are_on_same_side_of_wall(below_wall, below_wall)
    assert c._are_on_same_side_of_wall(above_wall, above_wall)
    assert not c._are_on_same_side_of_wall(below_wall, above_wall)


@pytest.fixture(params=[Labyrinth, ToyLab, HardLabyrinth, PandaEnv, PandaPickAndPlace], scope="module")
def env_fn(request):
    yield request.param


@pytest.fixture(params=["ToyLab-v0", "Labyrinth-v0", "HardLabyrinth-v0"], scope="module")
def env_gym_name(request):
    yield request.param


class TestSuiteForEnvs:
    def test_compute_reward(self, env_fn):
        env = env_fn()
        for _ in range(5):
            g = env.observation_space["desired_goal"].sample()
            assert env.compute_reward(g, g, None) == max(env.reward_range)

    def test_env_normalization(self, env_fn):
        if env_fn == PandaEnv:
            pytest.skip("Pandas Env can reach beyond -1, 1")

        env = env_fn()
        space = env.observation_space["desired_goal"]
        assert env.reward_range == (-1, 0)
        assert np.allclose(1, space.high[np.isfinite(space.high)])
        assert np.allclose(-1, space.low[np.isfinite(space.low)])

        env.reset()
        low = env.observation_space["achieved_goal"].low
        high = env.observation_space["achieved_goal"].high
        for _ in range(100):
            obs: Observation = env.step(env.action_space.high)[0]
            assert all(low <= obs.achieved_goal) and all(obs.achieved_goal <= high)

    def test_setting_goals_at_runtime(self, env_fn):
        env = env_fn()
        my_goals = [tuple(env.observation_space["desired_goal"].sample()) for _ in range(3)]
        for _ in range(3):
            assert tuple(env.reset().desired_goal) not in my_goals

        env.set_possible_goals(np.array(my_goals))
        for _ in range(3):
            obs = env.reset()
            assert any(np.allclose(obs.desired_goal, g) for g in my_goals), f"{obs.desired_goal} not in {my_goals}"

        env.set_possible_goals(None, entire_space=True)
        for _ in range(3):
            assert tuple(env.reset().desired_goal) not in my_goals

    @pytest.mark.parametrize("use_random_starting_pos", [True, False])
    def test_get_goal_successes(self, use_random_starting_pos: bool, env_fn):
        env = env_fn(use_random_starting_pos=use_random_starting_pos)
        assert all(len(successes) == 0 for successes in env.get_successes_of_goals().values())
        difficult_goal = env.observation_space["desired_goal"].high
        my_goals = np.array([env.starting_agent_pos, difficult_goal])
        env.set_possible_goals(my_goals)

        null_action = np.zeros(shape=env.action_space.shape)
        for _ in range(2):
            env.reset()
            for _ in range(3):
                env.step(null_action)

        successes = env.get_successes_of_goals()

        if not use_random_starting_pos:
            assert successes[tuple(env.starting_agent_pos)][0]
        assert len(successes[tuple(difficult_goal)]) == 0

    def test_moving_one_step_away_from_goal_still_success(self, env_fn):
        env = env_fn()
        env.set_possible_goals(env.starting_agent_pos[np.newaxis])
        env.reset()
        obs, r, done, info = env.step(env.action_space.high)
        assert np.allclose(obs.desired_goal, env.starting_agent_pos)
        assert info["is_success"] == 1
        assert env.compute_reward(obs.achieved_goal, obs.desired_goal, None) == env.reward_range[1]

    def test_seed_determines_trajectories(self, env_fn):
        null = np.zeros(shape=env_fn().action_space.shape)
        first_obs = env_fn(seed=0).step(null)[0]
        again = env_fn(seed=0).step(null)[0]
        assert first_obs == again
        assert env_fn(seed=0).step(null)[0] != env_fn(seed=1).step(null)[0]

        mk_actions = lambda env: [env.action_space.sample() for _ in range(10)]
        mk_obs = lambda env: [env.reset() for _ in range(10)]

        env = env_fn(seed=1)
        actions = mk_actions(env)
        obss = mk_obs(env)
        trajectory = [env.step(a) for a in actions]

        env2 = env_fn(seed=1)
        assert np.allclose(mk_actions(env2), actions)
        assert mk_obs(env2) == obss

        env.seed(1)
        env.reset()
        assert np.allclose(actions, mk_actions(env))
        new_obs = mk_obs(env)
        assert obss == new_obs
        assert trajectory == [env.step(a) for a in actions]

    def test_with_random_starting_states(self, env_fn):
        env = env_fn(use_random_starting_pos=True)
        o1: Observation
        starting_obss = [env.reset() for _ in range(5)]
        for o1, o2 in combinations(starting_obss, 2):
            assert not np.allclose(o1.achieved_goal, o2.achieved_goal)

        for obs in starting_obss:
            assert not np.allclose(obs.achieved_goal, obs.desired_goal)

    def test_gym_registration_succeded(self, env_gym_name):
        assert gym.make(env_gym_name) is not None, "The gym could not be loaded with gym.make." \
                                               "Check the env registration string."

    def test_multiple_envs_can_be_instantiated(self, env_fn):
        envs = [env_fn() for _ in range(3)]
        assert envs is not None

    def test_max_episode_len(self, env_fn):
        env = env_fn(max_episode_len=7)
        null_action = np.zeros(shape=env.action_space.shape)
        dones = [done(env.step(null_action)) for _ in range(6)]
        assert not dones[-1]
        assert done(env.step(null_action))

    def test_restart_reset_steps(self, env_fn):
        env = env_fn(max_episode_len=5)
        null_action = np.zeros(shape=env.action_space.shape)
        env.seed(0)
        while not done(env.step(null_action)):
            pass

        env.reset()
        env.set_possible_goals(env.observation_space["desired_goal"].high[None])
        assert not done(env.step(null_action))

    def test_env_trajectory(self, env_fn):
        env = env_fn(max_episode_len=10)
        agent = null_agent(action_space=env.action_space)
        assert len(list(trajectory(pi=agent, env=env))) == 10

        goal = env.observation_space["desired_goal"].high
        assert len(list(trajectory(pi=agent, env=env, goal=goal))) == 10

    def test_random_goals_cover_space(self, env_fn):
        env = env_fn(seed=1)
        reset_goals = np.array([env.reset().desired_goal for _ in range(100)])

        relevant_dims = np.isfinite(env.observation_space["desired_goal"].high)
        assert cover_space(reset_goals[:, relevant_dims])

    def test_obs_size_as_expected(self, env_fn_and_obs_size):
        env_fn, expected_obs_size = env_fn_and_obs_size
        env = env_fn()
        assert env.observation_space["observation"].shape[0] == expected_obs_size
        assert env.reset().observation.size == expected_obs_size
        null_action = np.zeros(shape=env.action_space.shape)
        assert env.step(null_action)[0].observation.size == expected_obs_size

    def test_parallel_envs_dont_affect_each_other(self, env_fn):
        env = env_fn()
        steady_obs = put_in_steady_state(env)

        env2 = env_fn()
        env2.reset()
        for _ in range(10):
            env2.step(env2.action_space.sample())[0]

        null_action = np.zeros(shape=env.action_space.shape)
        assert steady_obs.equal_except_time(env.step(null_action)[0])


def put_in_steady_state(env) -> Observation:
    """Meaning null actions dont keep observations the same"""
    null_action = np.zeros(shape=env.action_space.shape)
    old_obs = env.reset()
    while True:
        obs: Observation = env.step(null_action)[0]
        if obs.equal_except_time(old_obs):
            break
        old_obs = obs
    return obs


vel2d_plus_time = 3
time_only = 1
pos_and_vels = 2*7


@pytest.fixture(params=[(ToyLab, time_only),
                        (Labyrinth, vel2d_plus_time),
                        (HardLabyrinth, vel2d_plus_time),
                        (PandaEnv, pos_and_vels + time_only),
                        (PandaPickAndPlace, 3 + time_only)])
def env_fn_and_obs_size(request):
    return request.param


def done(env_res):
    return env_res[2]


def cover_space(samples: np.ndarray, tolerance=0.03) -> bool:
    return (np.allclose(samples.min(axis=0), -1, atol=tolerance) and
            np.allclose(samples.max(axis=0), 1, atol=tolerance))
