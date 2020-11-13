from gym.envs.registration import register

register(
    id='ToyLab-v0',
    entry_point='multi_goal.envs.toy_labyrinth_env:ToyLab',
)

register(
    id="Labyrinth-v0",
    entry_point="multi_goal.envs.pybullet_labyrinth_env:Labyrinth"
)

register(
    id="HardLabyrinth-v0",
    entry_point="multi_goal.envs.pybullet_labyrinth_env:HardLabyrinth"
)
