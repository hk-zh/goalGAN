from multi_goal.envs.pybullet_imgs import get_labyrinth_cam_settings
from multi_goal.envs.pybullet_labyrinth_env import Labyrinth

if __name__ == '__main__':
    env = Labyrinth(visualize=True)
    pb = env._sim._bullet
    import matplotlib.pyplot as plt
    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(**get_labyrinth_cam_settings(pb))
    plt.ion()
    plt.imshow(rgbImg)
    input("Exit")
