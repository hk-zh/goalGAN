from multi_goal.agents import PPOAgent, GoalGANAgent, HERSACAgent, EvaluateCallback
from multi_goal.envs.toy_labyrinth_env import ToyLab
import click


@click.command()
@click.option("--seed", required=True, type=int)
def cmd_main(*args, **kwargs):
    main(*args, **kwargs)


def main(seed: int):
    env = ToyLab(seed=seed)
    # π     = PPOAgent(env=env, experiment_name="goalgan-ppo-seed-{}".format(seed), rank=seed)
    π = HERSACAgent(env=env, experiment_name="goalgan-her-sac-{}".format(seed), rank=seed)
    agent = GoalGANAgent(env=env, agent=π)

    callback = EvaluateCallback(agent=agent, eval_env=ToyLab(seed=seed), rank=seed)
    agent.train(timesteps=int(1e6), callbacks=[callback])


if __name__ == '__main__':
    cmd_main()
