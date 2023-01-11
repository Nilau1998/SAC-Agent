from rendering.boat_env_render import BoatEnvironmentRenderer
from postprocessing.recorder import Recorder
from environment.boat_env import BoatEnv
from utils.config_reader import get_config, get_experiment_config
from utils.build_experiment import Experiment
from utils.plotting import plot_learning_curve
from utils.hyperparameter_tuner import HPTuner
from agent.continuous_agent import ContinuousAgent
import csv
import os
import numpy as np
import argparse
from progress_table import ProgressTable
import warnings

# Complains about tensors being transformed wrong and it being slow, it can be ignored after profiling showed it how insignificant it is
warnings.filterwarnings('ignore')


class ControlCenter:
    def __init__(self, subdir=None):
        self.config = get_config(os.path.join('original_config.yaml'))
        self.tuner = HPTuner('hp_configs.yaml')
        self.experiment_overview_file = os.path.join(
            'experiments', subdir, 'overview.csv')
        self.experiment = None
        self.subdir = subdir

    def train_model(self):
        self.experiment = Experiment(subdir=self.subdir)
        self.experiment.save_configs()

        env = BoatEnv(self.config, self.experiment)

        recorder = Recorder(env)
        recorder.write_winds_to_csv()

        agent = ContinuousAgent(
            config=self.config,
            experiment_dir=self.experiment.experiment_dir,
            input_dims=env.observation_space.shape,
            env=env
        )

        table_training = ProgressTable(
            columns=['Episode', 'Termination', 'Score',
                     'Best Score', 'Average Score', 'RA', 'Action RA'],
            num_decimal_places=2,
            default_column_width=14,
            reprint_header_every_n_rows=0,
        )

        best_score = 0
        score_history = []
        load_checkpoint = False

        if load_checkpoint:
            agent.load_models()

        for i in range(self.config.base_settings.n_games):
            table_training['Episode'] = i
            observation = env.reset()
            done = False
            score = 0

            recorder.create_csvs(i)
            table_training(self.config.boat.fuel)
            while not done:
                recorder.write_data_to_csv()
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.remember(observation, action, reward, observation_, done)
                if not load_checkpoint:
                    agent.learn()
                observation = observation_
                table_training.update('Score', score, weight=1)
                table_training.update(
                    'RA', f"{env.boat.rudder_angle:.2f}")
                table_training.update(
                    'Action RA', f"{env.action[0]:.2f}")

            recorder.write_info_to_csv()

            score_history.append(score)
            avg_score = np.mean(
                score_history[-self.config.base_settings.avg_lookback:])

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
        with open(os.path.join(self.experiment.experiment_dir, 'console.csv'), 'x') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerows(table_training.to_list())

        if not os.path.exists(self.experiment_overview_file):
            with open(self.experiment_overview_file, 'x') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow([self.experiment.experiment_name, best_score])
        else:
            with open(self.experiment_overview_file, 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow([self.experiment.experiment_name, best_score])

        if not load_checkpoint:
            x = [i+1 for i in range(self.config.base_settings.n_games)]
            figure_file = os.path.join(
                self.experiment.experiment_dir, 'plots', 'boat_env')
            plot_learning_curve(x, score_history, figure_file)

    def render_model(self, experiment_dir):
        if args.t:
            experiment_dir = self.experiment.experiment_dir
        else:
            experiment_dir = experiment_dir
        print(f"Starting rendering...")
        table_rendering = ProgressTable(
            columns=['Episode', 'Episodes left to render'],
            num_decimal_places=2,
            default_column_width=8,
            reprint_header_every_n_rows=0,
        )
        renderer = BoatEnvironmentRenderer(experiment_dir)

        relevant_episodes, best_episodes = renderer.replayer.analyse_experiment()
        episode_left = len(relevant_episodes) - 1
        for episode_index in relevant_episodes:
            table_rendering['Episode'] = episode_index
            table_rendering['Episodes left to render'] = episode_left
            renderer.replayer.read_data_csv(episode_index)
            if episode_index in best_episodes:
                renderer.previous_best_tmp = episode_index
            else:
                renderer.previous_best_tmp = -1
            for dt in table_rendering(range(renderer.replayer.total_dt)):
                table_rendering.next_row()
                if dt % self.config.base_settings.render_skip_size == 0 or dt == renderer.replayer.total_dt:
                    renderer.update_objects_on_image(episode_index, dt)
                    renderer.draw_image_to_buffer()
            renderer.create_gif_from_buffer(f"episode_{episode_index}")
            renderer.reset_renderer()
            episode_left -= 1
            table_rendering.next_row()
        table_rendering.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        action='store_true',
        help='train new model'
    )
    parser.add_argument(
        '-r',
        action='store_true',
        help='render trained model or parsed previous experiment dir'
    )
    parser.add_argument(
        'dir',
        nargs='?'
    )
    parser.add_argument(
        '-p',
        action='store_true',
        help='use hp tuner for model_generation'
    )
    parser.add_argument(
        'n_models',
        nargs='?'
    )
    args = parser.parse_args()

    control_center = ControlCenter(subdir='no_wind')
    if args.p:
        for _ in range(int(args.n_models)):
            control_center.tuner.set_config_file(control_center.config)
            control_center.train_model()
            if args.r:
                control_center.render_model()

    if args.t:
        control_center.train_model()
        if args.r:
            control_center.render_model(
                control_center.experiment.experiment_dir)
            args.r = False

    if args.r:
        control_center.render_model(args.dir)
