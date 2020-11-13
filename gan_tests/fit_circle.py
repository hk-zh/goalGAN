from itertools import count
from multi_goal.GenerativeGoalLearning import train_GAN
from multi_goal.LSGAN import LSGAN
from multi_goal.utils import get_updateable_scatter
import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

#### PARAMETERS ####
torch.set_default_dtype(torch.float64)

G_Input_Size  = 4
G_Hidden_Size = 128
D_Hidden_Size = 512

num_samples_goalGAN_goals  = 120
num_sample_random_goals    = num_samples_goalGAN_goals * 2
map_length = 1
eps = 0.1
Rmin = 0.3
Rmax = 0.4
target_point = torch.Tensor(np.random.random(size=(2)) * 2 * map_length - map_length) /2
####################


def label_goals_naive(samples, target):
    return [int(torch.dist(s, target) <= eps) for s in samples]

def label_goals_complex(samples, target):
    return [int(Rmin <= torch.dist(s, target) <= Rmax) for s in samples]

def init_plot(target, title: str = None):
    fig, ax, scatter_fn = get_updateable_scatter()
    ax.set_ylim(-map_length, map_length); ax.set_xlim(-map_length, map_length)
    if title is not None:
        ax.set_title(title)

    # plot target circles
    circle_rmin = plt.Circle(target, Rmin, color='green', alpha=0.1)
    circle_rmax = plt.Circle(target, Rmax, color='red',   alpha=0.1)
    ax.add_artist(circle_rmin)
    ax.add_artist(circle_rmax)

    def plot_goals(gan_goals, rand_goals):
        # plot gan_goals
        if gan_goals is not None:
            scatter_fn(name="gan_goals", pts=gan_goals.detach().numpy(), color='black')

        # plot rand_goals
        if rand_goals is not None:
            scatter_fn(name="rand_goals", pts=rand_goals.detach().numpy(), color="gray")

        fig.canvas.draw()
        fig.canvas.flush_events()

        return fig

    return plot_goals

def initial_gan_train(goalGAN):
    ## initial training of GAN with random samples
    ## aim is to make G generate evenly distributed goals before starting the actual training
    ## if we do not run this, G generates goals concentrated around (0,0)
    plot_fn = init_plot(target_point, title="Pretraining")

    for i in range(10):
        rand_goals  = torch.tensor(np.random.uniform(-1, 1, size=(num_samples_goalGAN_goals, 2)))        
        labels_rand = label_goals_complex(rand_goals, target_point)
        plot_fn(None, rand_goals)
        print("Init Iteration: {}".format(i))
        goalGAN   = train_GAN(rand_goals, labels_rand, goalGAN)

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    goalGAN = LSGAN(generator_input_size=G_Input_Size,
                        generator_hidden_size=G_Hidden_Size,
                        generator_output_size=2,
                        discriminator_input_size=2,
                        discriminator_hidden_size=D_Hidden_Size,
                        discriminator_output_size=1,
                        map_scale=map_length)

    initial_gan_train(goalGAN)
    plot_fn = init_plot(target_point, title="Training")

    ### training
    for i in range(50):
        z          = torch.randn(size=(num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
        gan_goals  = goalGAN.Generator.forward(z).detach()
        rand_goals = torch.tensor(np.random.uniform(-1, 1, size=(num_sample_random_goals,2)))
        goals      = torch.cat([gan_goals, rand_goals], axis=0)

        labels_gan  = label_goals_complex(gan_goals, target_point)
        labels_rand = label_goals_complex(rand_goals, target_point)
        labels      = labels_gan + labels_rand # concat lists 

        plot_fn(gan_goals, rand_goals)

        print("Iteration: {},   Number of generated positive samples: {}/{}".format(i, np.sum(labels_gan), gan_goals.shape[0]))
        if np.sum(labels) < 2:
            print(".. reinitializing GAN")
            goalGAN.reset_GAN()
            continue

        if np.sum(labels_gan) > gan_goals.shape[0] * 0.95:
            print(".. training done")
            break

        goalGAN   = train_GAN(goals, labels, goalGAN)

    ### validation
    plot_fn = init_plot(target_point, title="Validation")
    for i in count():
        z         = torch.randn(size=(num_samples_goalGAN_goals, goalGAN.Generator.noise_size))
        gan_goals = goalGAN.Generator.forward(z).detach()
        labels    = label_goals_complex(gan_goals, target_point)

        print("Number of generated positive samples: {}/{}".format(np.sum(labels), gan_goals.shape[0]))
        plot_fn(gan_goals, None)
        input("Press any key for next iter")

if __name__ == '__main__':
    main()
