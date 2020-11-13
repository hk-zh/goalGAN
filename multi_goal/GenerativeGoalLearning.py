import time
import warnings
from itertools import count
from typing import Sequence, Tuple, Iterator

import gym
import numpy as np
import torch
from more_itertools import consume
from torch import Tensor

from multi_goal.envs import Observation, ISettableGoalEnv, dim_goal
from multi_goal.LSGAN import LSGAN
from multi_goal.utils import print_message, display_goals

#### PARAMETERS ####
Rmin = 0.1
Rmax = 0.9

G_Input_Size  = 4       # noise dim, somehow noise size is defined as 4 in their implementation for ant_gan experiment
G_Hidden_Size = 32
D_Hidden_Size = 64
GEN_VAR_COEFF = 0.001
####################

Returns = Sequence[float]


class Agent:
    name: str

    def __call__(self, obs: Observation) -> np.ndarray:
        raise NotImplementedError

    def train(self, timesteps: int) -> None:
        raise NotImplementedError


def random_agent(action_space: gym.spaces.Space) -> Agent:
    return lambda obs: action_space.sample()


def null_agent(action_space: gym.spaces.Space) -> Agent:
    return lambda obs: np.zeros(shape=action_space.shape)


def sample(t: Tensor, k: int) -> Tensor:
    """
    https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
    Implemented according to Appendix A.1: 2/3 from gan generated goals, 1/3 from old goals
    TODO: To avoid concentration of goals, concatinate only the goals which are away from old_goals
    """
    num_samples = min(len(t), k)
    indices = torch.randperm(len(t))[:num_samples]
    return t[indices]


def initialize_GAN(env: gym.GoalEnv) -> LSGAN:
    goalGAN = LSGAN(generator_input_size=G_Input_Size,
                    generator_hidden_size=G_Hidden_Size,
                    generator_output_size=dim_goal(env),
                    discriminator_input_size=dim_goal(env),
                    discriminator_hidden_size=D_Hidden_Size,
                    gen_variance_coeff=GEN_VAR_COEFF,
                    discriminator_output_size=1) # distinguish whether g is in GOID or not
    return goalGAN


def update_and_eval_policy(goals: Tensor, π: Agent, env: ISettableGoalEnv) -> Tuple[Agent, Returns]:
    env.set_possible_goals(goals.numpy())
    env.reset()

    print("Training the policy on current goals...", end=" ", flush=True)
    start = time.time()
    while True:
        episode_successes_per_goal = env.get_successes_of_goals()
        if all(len(sucs) >= 3 for g, sucs in episode_successes_per_goal.items()):
            print(f"DONE. TIME: {(time.time() - start):.2f} [s]", flush=True)
            break
        yield

    returns = [np.mean(episode_successes_per_goal[tuple(g)]) for g in goals.numpy()]
    return π, returns


def eval_policy(goals: Tensor, π: Agent, env: ISettableGoalEnv):
    for g in goals.numpy():
        for obs, action, reward, next_obs, done, info in trajectory(π, env, goal=g):
            if info.get("is_success"):
                break
    episode_successes_per_goal = env.get_successes_of_goals()
    returns = [np.mean(episode_successes_per_goal[tuple(g)]) for g in goals.numpy()]
    return returns


def label_goals(returns: Returns) -> Sequence[int]:
    return [int(Rmin <= r <= Rmax) for r in returns]


@print_message("Training GAN on current goals")
def train_GAN(goals: Tensor, labels: Sequence[int], goalGAN):
    y: Tensor = torch.Tensor(labels).reshape(len(labels), 1)
    D = goalGAN.Discriminator.forward
    G = goalGAN.Generator.forward

    def D_loss_vec(z: Tensor) -> Tensor:
        return y*(D(goals)-1)**2 + (1-y)*(D(goals)+1)**2 +(D(G(z))+1)**2

    iterations = 10
    for _ in range(iterations):
        ### Train Discriminator ###
        gradient_steps = 1
        for _ in range(gradient_steps):
            zs = torch.randn(len(labels), goalGAN.Generator.noise_size)
            goalGAN.Discriminator.zero_grad()
            D_loss = torch.mean(D_loss_vec(zs))
            D_loss.backward()
            goalGAN.D_Optimizer.step()

        ### Train Generator ###
        gradient_steps = 1
        β = goalGAN.Generator.variance_coeff
        for _ in range(gradient_steps):
            zs = torch.randn(len(labels), goalGAN.Generator.noise_size)
            goalGAN.Generator.zero_grad()
            G_loss = torch.mean(D(G(zs))**2) + β/torch.var(G(zs), dim=0).mean()
            G_loss.backward()
            goalGAN.G_Optimizer.step()

    return goalGAN


@print_message("Updating the regularized replay buffer")
def update_replay(goals: Tensor, goals_old: Tensor):
    if goals_old.shape[0] == 0:
        goals_old = goals[0][None]

    eps = 0.1
    for g in goals:
        g_is_close_to_goals_old = min((torch.dist(g, g_old) for g_old in goals_old)) < eps
        if not g_is_close_to_goals_old:
            goals_old = torch.cat((g[None], goals_old))
    return goals_old


def trajectory(pi: Agent, env: ISettableGoalEnv, goal: np.ndarray = None,
               sleep_secs: float = 0, render=False, print_every: int = None):
    if goal is not None:
        env.set_possible_goals(np.array(goal)[np.newaxis])
    obs = env.reset()

    for t in count():
        action = pi(obs)
        next_obs, reward, done, info = env.step(action)

        if sleep_secs > 0:
            time.sleep(sleep_secs)
        if render:
            env.render()

        if print_every is not None and t % print_every == 0:
            print(f"achieved goal: {obs.achieved_goal.T},"
                  f" desired goal: {obs.desired_goal.T}")
        if print_every is not None and info.get("is_success"):
            print(f"SUCCESS! Episode len: {t}")

        yield obs, action, reward, next_obs, done, info

        obs = next_obs
        if done:
            if print_every is not None and info.get("is_success") == 0:
                print(f"FAILURE! Episode len: {t}")
            break


@print_message("Evaluating agent in env...")
def evaluate(agent: Agent, env: ISettableGoalEnv, very_granular=False, plot=True, silent=False,
             coarseness_per_dim=30):
    coarseness = complex(0, coarseness_per_dim if very_granular else 6)
    goal_dim = env.observation_space["desired_goal"].shape[0]
    if goal_dim == 2:
        goals = np.mgrid[-1:1:coarseness, -1:1:coarseness].reshape((2, -1)).T
    else:
        goals = np.array([env.observation_space["desired_goal"].sample() for _ in range(50)])

    env.set_possible_goals(goals)
    for idx, _ in enumerate(goals):
        consume(trajectory(agent, env))
        if not silent:
            print(f"done evaluation goal #{idx}", flush=True)
    reached = np.array([goal for goal, successes in env.get_successes_of_goals().items() if successes[0]])
    not_reached = np.array([goal for goal, successes in env.get_successes_of_goals().items() if not successes[0]])
    env.set_possible_goals(goals=None, entire_space=True)
    if plot:
        env.render(other_positions={"red": not_reached, "green": reached},
                   show_agent_and_goal_pos=False)
    return reached, not_reached


def train_goalGAN(π: Agent, goalGAN: LSGAN, env: ISettableGoalEnv, pretrain_iters=5, use_old_goals=True) -> Iterator[None]:
    """
    Algorithm in the GAN paper, Florensa 2018

    for i in iterations:
        z         = sample_noise()                     # input for goal generator network
        goals     = G(z) union goals_old               # concat old goals with the generated ones
        π         = update_policy(goals, π)            # perform policy update, paper uses TRPO, Leon suggested to use PPO as it is simpler
        returns   = evaluate_policy(goals, π)          # needed to label the goals
        labels    = label_goals(goals)                 # needed to train discriminator network
        G, D      = train_GAN(goals, labels, G, D)
        goals_old = goals

    """

    #### PARAMETERS ####
    num_gan_goals       = 60
    num_old_goals       = num_gan_goals // 2 if use_old_goals else 0
    num_rand_goals      = num_gan_goals // 2
    ####################

    log_iter = lambda it_num: print(f"\n### BEGIN ITERATION {it_num} ###")

    # Initial training of the policy with random goals
    for iter_num in range(pretrain_iters):
        log_iter(iter_num)
        rand_goals = torch.clamp(torch.Tensor([env.starting_agent_pos]) + 0.1*torch.randn(num_old_goals, dim_goal(env)), min=-1, max=1)
        π, returns = yield from update_and_eval_policy(rand_goals, π, env)
        labels     = label_goals(returns)
        display_goals(rand_goals.detach().numpy(), returns, iter_num, env, fileNamePrefix='_')
        goalGAN    = train_GAN(rand_goals, labels, goalGAN)

    close_to_starting_pos = torch.Tensor([env.starting_agent_pos]) + 0.1*torch.randn(num_old_goals, dim_goal(env))
    goals_old = torch.clamp(close_to_starting_pos, min=-1, max=1)

    for iter_num in count(pretrain_iters):
        log_iter(iter_num)
        z             = torch.randn(size=(num_gan_goals, goalGAN.Generator.noise_size))
        raw_gan_goals = goalGAN.Generator.forward(z).detach()
        gan_goals     = torch.clamp(raw_gan_goals + 0.2*torch.randn(num_gan_goals, dim_goal(env)), min=-1, max=1)
        rand_goals    = torch.Tensor(num_rand_goals, dim_goal(env)).uniform_(-1, 1)
        all_goals     = torch.cat([gan_goals, sample(goals_old, k=num_old_goals), rand_goals])
        π, returns    = yield from update_and_eval_policy(all_goals, π, env)
        display_goals(all_goals.detach().numpy(), returns, iter_num, env, gan_goals=raw_gan_goals.numpy())
        labels        = label_goals(returns)
        if all([lab == 0 for lab in labels]): warnings.warn("All labels are 0")
        goalGAN       = train_GAN(all_goals, labels, goalGAN)
        goals_old     = update_replay(gan_goals, goals_old=goals_old)
