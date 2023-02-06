import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

if __name__ == '__main__':
    file = 'data.csv'
    data = pd.read_csv(file, sep=';')

    sns.set_style("whitegrid", {'axes.grid': True,
                                'axes.edgecolor': 'black'})
    sns.histplot(data['reached_goal_4'], element="step",
                 alpha=0.5, color='b')  # , color='b'
    sns.histplot(data['reached_goal_5'], element="step",
                 alpha=0.5, color='c')  # , color='c'
    sns.histplot(data['reached_goal_6'], element="step",
                 alpha=0.5, color='r')  # , color='r'
    sns.despine()
    plt.xlabel('Ziel erreicht')
    plt.ylabel('Anzahl der Ergebnisse')
    plt.title(
        'Verteilung der Episodenenden "Ziel erreicht" über \n die einzelnen Ergebnisse')
    labels = ['Experiment 4', 'Experiment 5', 'Experiment 6']
    plt.legend(labels=labels)
    plt.savefig('bla.png', dpi=300)
    plt.close('all')
