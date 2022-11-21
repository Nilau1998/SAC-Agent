import os
import numpy as np
from agent.continuous_agent import ContinuousAgent
from utils.plotting import plot_learning_curve
from utils.build_experiment import Experiment
from utils.config_reader import get_config
from environments.boat_env import BoatEnv
from rendering.env_render import EnvironmentRenderer

if __name__ == '__main__':
    experiment = Experiment()
    experiment.save_configs()
    config = get_config(os.path.join("config.yaml"))

    environment = "boat_env"

    env = BoatEnv(config, experiment)

    env_render = EnvironmentRenderer(config, env)

    agent = ContinuousAgent(
        config=config,
        experiment_dir=experiment.experiment_dir,
        input_dims=env.observation_space.shape,
        env=env
    )

    n_games = config.base_settings.n_games
    filename = environment

    figure_file = os.path.join(experiment.experiment_dir, "plots", filename)

    best_score = 0
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0

        while not done:
            env_render.create_new_image()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        if score > best_score:
            best_score = score
            env_render.set_best_path()
            env_render.create_gif_from_buffer(os.path.join(experiment.experiment_dir, "rendering"), f"episode_{i}")
            env_render.reset_renderer()
            if not load_checkpoint:
                agent.save_models()

        env_render.reset_renderer()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print(
            f"episode: {i}, "
            f"score: {score:.1f}, "
            f"best score: {best_score:.1f}, "
            f"avg_score: {avg_score:.1f}"
        )
        print(info, "\n")

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)