import click
from more_itertools import consume

from multi_goal.GenerativeGoalLearning import trajectory
from multi_goal.agents import HERSACAgent, EvaluateCallback, GoalGANAgent,PPOAgent
from multi_goal.envs.pybullet_panda_robot import PandaEnv, PandaPickAndPlace


class Task:
    REACH = "reach"
    PICK_AND_PLACE = "pickplace"


@click.command()
@click.option("--task", type=click.Choice([Task.REACH, Task.PICK_AND_PLACE]))
@click.option("--use-gan", is_flag=True, default=False)
@click.option("--do-train", is_flag=True, default=False)
@click.option("--perform-eval", is_flag=True, default=False)
def main(task: str, use_gan: bool, do_train: bool, perform_eval: bool):
    env_params = {"visualize": not do_train}
    env_fn = PandaEnv if task == Task.REACH else PandaPickAndPlace
    env = env_fn(**env_params)

    agent_params = {"env": env}
    if use_gan:
        agent_params["experiment_name"] = "goalgan-her-sac"
    agent = HERSACAgent(**agent_params)
    if use_gan:
        agent = GoalGANAgent(env=env, agent=agent)

    if do_train:
        cbs = [EvaluateCallback(agent=agent, eval_env=env_fn(**env_params))] if perform_eval else []
        agent.train(timesteps=50000, callbacks=cbs)
    else:
        while True:
            consume(trajectory(agent, env))


def continuous_viz():
    env = PandaEnv(visualize=True)
    agent = HERSACAgent(env=env, experiment_name="goalgan-her-sac")
    obs = env.reset()
    while True:
        action = agent(obs)
        obs, _, done, info = env.step(action)
        if done:
            obs = env.reset(reset_agent_pos=not info["is_success"])


if __name__ == '__main__':
    main()
