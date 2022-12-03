import os
import numpy as np
import argparse

from agent.continuous_agent import ContinuousAgent
from utils.plotting import plot_learning_curve
from utils.build_experiment import Experiment
from utils.config_reader import get_config
from environment.boat_env import BoatEnv
from postprocessing.recorder import Recorder
from rendering.boat_env_render import BoatEnvironmentRenderer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        action='store_true',
        help='turn on renderer after learning'
    )
    args = parser.parse_args()

    experiment = Experiment()
    experiment.save_configs()
    config = get_config(os.path.join('config.yaml'))

    environment = 'boat_env'

    env = BoatEnv(config, experiment)

    recorder = Recorder(env)
    recorder.write_winds_to_csv()

    agent = ContinuousAgent(
        config=config,
        experiment_dir=experiment.experiment_dir,
        input_dims=env.observation_space.shape,
        env=env
    )

    n_games = config.base_settings.n_games
    filename = environment

    best_score = 0
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0

        recorder.create_csvs(i)

        while not done:
            recorder.write_data_to_csv()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        recorder.write_info_to_csv()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if score > best_score:
            best_score = score

        if score > avg_score:
            if not load_checkpoint:
                agent.save_models()

        print(
            f"episode: {i}, "
            f"score: {score:.1f}, "
            f"best score: {best_score:.1f}, "
            f"avg_score: {avg_score:.1f}"
        )
        print(info, "\n")

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        figure_file = os.path.join(
            experiment.experiment_dir, 'plots', filename)
        plot_learning_curve(x, score_history, figure_file)

    if args.r:
        print(f"Starting rendering...")
        renderer = BoatEnvironmentRenderer(experiment.experiment_dir)
        for episode_index in range(renderer.replayer.total_episodes):
            renderer.replayer.read_data_csv(episode_index)
            for dt in range(renderer.replayer.total_dt):
                renderer.update_objects_on_image(episode_index, dt)
                renderer.draw_image_to_buffer()
            renderer.create_gif_from_buffer(f"episode_{episode_index}")
            renderer.reset_renderer()
