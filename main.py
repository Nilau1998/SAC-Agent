from rendering.boat_env_render import BoatEnvironmentRenderer
from postprocessing.recorder import Recorder
from environment.boat_env import BoatEnv
from utils.config_reader import get_config
from utils.build_experiment import Experiment
from utils.plotting import plot_learning_curve
from agent.continuous_agent import ContinuousAgent
import os
import numpy as np
import argparse
from progress_table import ProgressTable
import sys
import warnings

warnings.filterwarnings("ignore")


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

    table_training = ProgressTable(
        columns=['Episode', 'Termination', 'Score',
                 'Best Score', 'Average Score'],

        # Default arguments:
        refresh_rate=10,
        num_decimal_places=2,
        default_column_width=8,
        default_column_alignment='center',
        print_row_on_update=True,
        reprint_header_every_n_rows=0,
        custom_format=None,
        embedded_progress_bar=False,
        table_style='normal',
        file=sys.stdout,
    )

    n_games = config.base_settings.n_games
    filename = environment

    best_score = 0
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        table_training['Episode'] = i
        observation = env.reset()[0]
        done = False
        score = 0

        recorder.create_csvs(i)
        table_training(config.boat.fuel)
        while not done:
            table_training.update('Score', score, weight=1)
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

        table_training['Termination'] = f"{info['termination']}-{info[info['termination']]}"
        table_training['Score'] = score
        table_training['Best Score'] = best_score
        table_training['Average Score'] = avg_score
        table_training.next_row()
    table_training.close()

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        figure_file = os.path.join(
            experiment.experiment_dir, 'plots', filename)
        plot_learning_curve(x, score_history, figure_file)

    if args.r:
        print(f"Starting rendering...")

        table_rendering = ProgressTable(
            columns=['Episode', 'Episodes left to render'],

            # Default arguments:
            refresh_rate=10,
            num_decimal_places=2,
            default_column_width=8,
            default_column_alignment='center',
            print_row_on_update=True,
            reprint_header_every_n_rows=0,
            custom_format=None,
            embedded_progress_bar=False,
            table_style='normal',
            file=sys.stdout,
        )
        renderer = BoatEnvironmentRenderer(experiment.experiment_dir)

        relevant_episodes = renderer.replayer.analyse_experiment()
        episode_left = len(relevant_episodes) - 1
        for episode_index in relevant_episodes:
            table_rendering['Episode'] = episode_index
            table_rendering['Episodes left to render'] = episode_left
            renderer.replayer.read_data_csv(episode_index)
            for dt in table_rendering(range(renderer.replayer.total_dt)):
                table_rendering.next_row()
                if dt % config.base_settings.render_skip_size == 0 or dt == renderer.replayer.total_dt:
                    renderer.update_objects_on_image(episode_index, dt)
                    renderer.draw_image_to_buffer()
            renderer.create_gif_from_buffer(f"episode_{episode_index}")
            renderer.reset_renderer()
            episode_left -= 1
            table_rendering.next_row()
        table_rendering.close()
