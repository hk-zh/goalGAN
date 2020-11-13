from itertools import product

from multi_goal.agents import HERSACAgent, EvaluateCallback, GoalGANAgent,PPOAgent
from multi_goal.envs.pybullet_panda_robot import PandaEnv
import multiprocessing


def run_exp(inputs):
    seed, use_goalgan = inputs
    env = PandaEnv(seed=seed)
    name = f"{'goalgan-' if use_goalgan else ''}her-sac"
    agent = HERSACAgent(env=env, rank=seed, experiment_name=name)
    if use_goalgan:
        agent = GoalGANAgent(env=env, agent=agent)
    cb = EvaluateCallback(agent=agent, eval_env=PandaEnv(seed=seed), rank=seed)
    agent.train(timesteps=int(50000), callbacks=[cb])


if __name__ == '__main__':
    seeds = range(10)
    use_goalgan = [False]
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(func=run_exp, iterable=product(seeds, use_goalgan))
