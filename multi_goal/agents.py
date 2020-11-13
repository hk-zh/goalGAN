import os
import warnings
from typing import Sequence, Callable

import numpy as np
from stable_baselines.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.her import HERGoalEnvWrapper
from torchsummary import summary

from multi_goal.utils import Dirs

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    from stable_baselines import PPO2, HER, SAC
from multi_goal.GenerativeGoalLearning import Agent, evaluate, train_goalGAN, initialize_GAN, \
    trajectory
from multi_goal.envs import Observation, ISettableGoalEnv, dim_goal


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, env: ISettableGoalEnv, verbose=1, experiment_name="ppo", rank=0):
        self._env = env
        self._dirs = Dirs(experiment_name=f"{type(env).__name__}-{experiment_name}", rank=rank)
        self._flat_env = HERGoalEnvWrapper(env)
        options = {"env": DummyVecEnv([lambda: self._flat_env]), "tensorboard_log": self._dirs.tensorboard,
                   "gamma": 1, "seed": rank, "nminibatches": 1}
        if os.path.isdir(self._dirs.models) and os.path.isfile(self._dirs.best_model):
            self._model = PPO2.load(load_path=self._dirs.best_model, **options)
            print(f"Loaded model {self._dirs.best_model}")
        else:
            self._model = PPO2("MlpPolicy", verbose=verbose, **options)

    def __call__(self, obs: Observation) -> np.ndarray:
        flat_obs = self._flat_env.convert_dict_to_obs(obs)
        action, _ = self._model.predict(flat_obs, deterministic=True)
        return action

    def train(self, timesteps: int, num_checkpoints=4, callbacks: Sequence[BaseCallback] = None):
        ppo_offset = 128
        callbacks = [] if callbacks is None else callbacks
        cb = CheckpointCallback(save_freq=timesteps//num_checkpoints, save_path=self._dirs.models, name_prefix=self._dirs.prefix)
        self._model.learn(total_timesteps=timesteps+ppo_offset, callback=CallbackList([cb, *callbacks]), log_interval=100)


class HERSACAgent(Agent):
    name = "her-sac"

    def __init__(self, env: ISettableGoalEnv, verbose=1, rank=0, experiment_name="her-sac"):
        self._env = env
        self._dirs = Dirs(experiment_name=f"{type(env).__name__}-{experiment_name}", rank=rank)
        options = {"env": env, "tensorboard_log": self._dirs.tensorboard, "model_class": SAC,
                   "gamma": 1, "learning_rate": 3e-3}
        if os.path.isdir(self._dirs.models) and os.path.isfile(self._dirs.best_model):
            self._model = HER.load(load_path=self._dirs.best_model, **options)
            print(f"Loaded model {self._dirs.best_model}")
        else:
            self._model = HER(policy="MlpPolicy", verbose=verbose, **options)

    def __call__(self, obs: Observation) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action

    def train(self, timesteps: int, callbacks: Sequence[BaseCallback] = None, num_checkpoints=4) -> None:
        callbacks = [] if callbacks is None else callbacks
        cb = CheckpointCallback(save_freq=timesteps//num_checkpoints, save_path=self._dirs.models, name_prefix=self._dirs.prefix)
        self._model.learn(total_timesteps=timesteps, callback=CallbackList([cb, *callbacks]))


class EvaluateCallback(BaseCallback):
    def __init__(self, agent: Agent, eval_env: ISettableGoalEnv, rank=0, specific_goal: np.ndarray = None,
                 experiment_name: str = None):
        super().__init__()
        self._specific_goal = specific_goal
        metric = "MapPctCovered" if specific_goal is None else "CanReachTargetGoal"
        self._agent = agent
        self._eval_env = eval_env
        experiment_name = agent.name if experiment_name is None else experiment_name
        self._log_fname = f"{type(eval_env).__name__}-{experiment_name}-{rank}-performance.csv"
        with open(self._log_fname, "w") as file:
            file.write(f"Step,{metric}\n")

    def _on_step(self) -> bool:
        if self.num_timesteps % 500 == 0:
            if self._specific_goal is None:
                self._log_full_goal_space_performance()
            else:
                self._log_specific_goal_performance()
        return True

    def _log_full_goal_space_performance(self):
        reached, not_reached = evaluate(agent=self._agent, env=self._eval_env, plot=False,
                                        silent=True, very_granular=True, coarseness_per_dim=10)
        pct = len(reached) / (len(reached) + len(not_reached))
        log = f"{self.num_timesteps},{round(pct, 6)}\n"
        with open(self._log_fname, "a") as file:
            file.write(log)

    def _log_specific_goal_performance(self):
        for step in trajectory(pi=self._agent, env=self._eval_env, goal=self._specific_goal):
            pass
        info = step[-1]
        log = f"{self.num_timesteps},{int(info['is_success'])}\n"
        with open(self._log_fname, "a") as file:
            file.write(log)


class GoalGANAgent(Agent):
    def __init__(self, env: ISettableGoalEnv, agent: Agent):
        self.name = f"goalgan-{agent.name}"
        self._agent = agent
        self._gan = initialize_GAN(env=env)
        self._env = env
        summary(self._gan.Generator,     input_size=(1, 1, 4), device='cpu')
        summary(self._gan.Discriminator, input_size=(1, 1, dim_goal(env)), device='cpu')

    def __call__(self, obs: Observation) -> np.ndarray:
        return self._agent(obs)

    def train(self, timesteps: int, use_buffer=True, callbacks: Sequence[BaseCallback] = None) -> None:
        loop = train_goalGAN(π=self._agent, goalGAN=self._gan, env=self._env,
                             use_old_goals=use_buffer)

        callbacks = [] if callbacks is None else callbacks
        cb = AnyFunctionTrainingCallback(callback=lambda: next(loop))
        self._agent.train(timesteps=timesteps, callbacks=[cb, *callbacks])


class AnyFunctionTrainingCallback(BaseCallback):
    def __init__(self, callback: Callable[[], None]):
        super().__init__()
        self._callback = callback

    def _on_rollout_start(self) -> None:
        self._callback()


class RandomAgent(Agent):
    def __init__(self, env: ISettableGoalEnv):
        self._action_space = env.action_space

    def __call__(self, obs: Observation) -> np.ndarray:
        return self._action_space.sample()

    def train(self, timesteps: int) -> None:
        pass
