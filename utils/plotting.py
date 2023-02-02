import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_learning_curve(scores, rng_avg, figure_file):
    plt.close('all')
    if not os.path.exists(figure_file):
        sns.set_style("whitegrid", {'axes.grid': True,
                                    'axes.edgecolor': 'black'})
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-rng_avg):(i+1)])
        sns.lineplot(data=running_avg, ci=95)
        # plt.plot(range(len(scores)), running_avg)
        plt.xlabel('Steps')
        plt.ylabel('Episoden Reward Mittelwert')
        plt.title('Gleitender Mittelwert über 50 Episoden des Rewards')
        sns.despine()
        plt.savefig(figure_file, dpi=300)
        plt.close('all')


def plot_stacked_area(experiment_dir):
    data = os.path.join(experiment_dir, 'episodes', 'info.csv')
    plt.close('all')
    data = pd.read_csv(data, sep=';')
    data = data.drop(['termination', 'episode_reward'], axis=1)
    new_columns = data.columns[data.loc[data.last_valid_index()].argsort()]
    data = data[new_columns]

    # color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    sns.set_style("whitegrid", {'axes.grid': True,
                                'axes.edgecolor': 'black',
                                "patch.force_edgecolor": False})
    plt.stackplot(data.index, data.values.T,
                  labels=data.keys())
    plt.xlabel('Episode')
    plt.ylabel('Häufigkeit einer Beendung')
    plt.title('Häufigkeit der Episodenenden über die Episoden')
    plt.legend(loc='upper left')
    sns.despine()
    plt.savefig(os.path.join(experiment_dir, 'stacked_plot.png'), dpi=300)
    plt.close('all')
