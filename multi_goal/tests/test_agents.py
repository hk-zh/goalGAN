from more_itertools import consume

from multi_goal.GenerativeGoalLearning import trajectory
from multi_goal.agents import PPOAgent, HERSACAgent, EvaluateCallback, GoalGANAgent
from multi_goal.envs.toy_labyrinth_env import ToyLab


def test_class_instantiation():
    env = ToyLab()
    a1 = PPOAgent(env=env)
    a2 = HERSACAgent(env=env)
    a3 = GoalGANAgent(env=env, agent=a1)
    for a in [a1, a2, a3]:
        consume(trajectory(pi=a, env=env))

    cb = EvaluateCallback(agent=a1, eval_env=env)
