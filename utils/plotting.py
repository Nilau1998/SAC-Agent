import numpy as np
import matplotlib.pyplot as plt
import os


def plot_learning_curve(scores, rng_avg, figure_file):
    plt.close('all')
    if not os.path.exists(figure_file):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-rng_avg):(i+1)])
        plt.plot(range(len(scores)), running_avg)
        plt.title('Running average of previous 50 scores')
        plt.savefig(figure_file)
        plt.close('all')
