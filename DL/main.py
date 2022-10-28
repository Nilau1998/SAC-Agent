import os
import gym
import numpy as np
import pybullet_envs
from gym.spaces import Box
from agent.continuous_agent import ContinuousAgent
from utils.plot_learning_curve import plot_learning_curve
from utils.build_experiment import Experiment
from utils.config_reader import get_config
from environments.shower_env import ShowerEnv

if __name__ == '__main__':
    experiment = Experiment()
    experiment.save_configs()
    config = get_config(os.path.join("config.yaml"))

    # environment = "MountainCarContinuous-v0"
    # environment = "InvertedPendulumBulletEnv-v0"
    environment = "MountainCar-v0"

    # env = gym.make(environment)

    env = ShowerEnv()

    assert env.action_space == Box, "Action space must be a Box (Continious), Discrete spaces are not implemented yet!"
    agent = ContinuousAgent(
        config=config,
        experiment_dir=experiment.experiment_dir,
        input_dims=env.observation_space.shape,
        env=env
    )

    n_games = config.base_settings.n_games
    filename = environment

    figure_file = os.path.join(experiment.experiment_dir, "plots", filename)

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)