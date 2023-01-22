from rendering.boat_env_render import BoatEnvironmentRenderer
from postprocessing.recorder import Recorder
from environment.boat_env import BoatEnv
from utils.config_reader import get_config
from utils.build_experiment import Experiment
from utils.hyperparameter_tuner import HPTuner
from utils.color_selector import ColorSelector
from agent.continuous_agent import ContinuousAgent
from multiprocessing import Process
import time
import csv
import os
import numpy as np
import argparse
from progress_table import ProgressTable
import warnings

# Complains about tensors being transformed wrong and it being slow, it can be ignored after profiling showed it how insignificant it is
warnings.filterwarnings('ignore')


class ControlCenter:
    def __init__(self, cc_id=0, color='\x1b[37m', subdir=None, config='original_config.yaml'):
        self.cc_id = cc_id
        self.color = color
        self.config = get_config(os.path.join(config))
        self.tuner = HPTuner('hp_configs.yaml')
        self.experiment_overview_file = os.path.join(
            'experiments', subdir, 'overview.csv')
        self.terminations_file = None
        self.info = None
        self.experiment = None
        self.subdir = subdir

    def train_model(self):
        self.experiment = Experiment(subdir=self.subdir)
        self.experiment.save_configs()
        self.terminations_file = os.path.join(
            self.experiment.experiment_dir, 'terminations.csv')

        env = BoatEnv(self.config, self.experiment)

        recorder = Recorder(env)

        agent = ContinuousAgent(
            config=self.config,
            experiment_dir=self.experiment.experiment_dir,
            input_dims=env.observation_space.shape,
            env=env
        )
        columns = ['CCID Episode', 'Termination', 'Score',
                   'Best Score', 'Average Score', 'RA', 'Action RA']
        table_training = ProgressTable(
            columns=columns,
            num_decimal_places=2,
            default_column_width=14,
            reprint_header_every_n_rows=0,
        )
        for column in table_training.columns:
            table_training._colors[column] = self.color
        best_score = float('-inf')
        score_history = []
        load_checkpoint = False

        if load_checkpoint:
            agent.load_models()

        for i in range(self.config.base_settings.n_games):
            table_training['CCID Episode'] = (self.cc_id, i)
            observation = env.reset()
            done = False
            score = 0

            recorder.create_csvs(i)
            table_training(self.config.boat.fuel)
            while not done:
                recorder.write_data_to_csv()
                action = agent.choose_action(observation)
                observation_, reward, done, self.info = env.step(action)
                score += reward
                if self.info['termination'] == 'reached_goal':
                    agent.remember(observation, action,
                                   reward, observation_, True)
                else:
                    agent.remember(observation, action,
                                   reward, observation_, False)
                if not load_checkpoint:
                    agent.learn()
                observation = observation_
                table_training.update('Score', score, weight=1)
                table_training.update(
                    'RA', f"{env.boat.rudder_angle:.2f}")
                table_training.update(
                    'Action RA', f"{env.action[0]:.2f}")
            recorder.write_info_to_csv()
            recorder.write_winds_to_csv()
            score_history.append(score)
            avg_score = np.mean(
                score_history[-self.config.base_settings.avg_lookback:])

            if score > best_score:
                best_score = score

            if score > avg_score:
                if not load_checkpoint:
                    agent.save_models()

            table_training['Termination'] = f"{self.info['termination']}-{self.info[self.info['termination']]}"
            table_training['Score'] = score
            table_training['Best Score'] = best_score
            table_training['Average Score'] = avg_score
            table_training.next_row()
        table_training.close()
        with open(os.path.join(self.experiment.experiment_dir, 'console.csv'), 'x') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(columns)
            writer.writerows(table_training.to_list())

        if not os.path.exists(self.experiment_overview_file):
            with open(self.experiment_overview_file, 'x') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow([self.experiment.experiment_name, best_score])
        else:
            with open(self.experiment_overview_file, 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow([self.experiment.experiment_name, best_score])

        with open(self.terminations_file, 'x') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(self.info.keys())
            writer.writerow(self.info.values())

        print(f"Process {self.cc_id} finished the training!")

    def render_model(self, experiment_dir):
        if args['train']:
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
            renderer.create_gif_from_buffer(episode_index)
            renderer.reset_renderer()
            episode_left -= 1
            table_rendering.next_row()
        table_rendering.close()

    def train_hp_model(self):
        self.tuner.set_config_file(control_center.config)
        self.train_model()

    def creates_avg_plots(self, directory):
        list_experiment_dirs = [
            f.path for f in os.scandir(directory) if f.is_dir()]
        for experiment_dir in list_experiment_dirs:
            BoatEnvironmentRenderer(experiment_dir)


if __name__ == '__main__':
    subdir = 'constant_wind_upwards'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--train',
        nargs='?',
        const='original_config.yaml',
        type=str,
        help='train new model with default or parsed config file'
    )
    parser.add_argument(
        '-r', '--render',
        nargs='?',
        const='',
        type=str,
        help='render trained model or parsed previous experiment dir'
    )
    parser.add_argument(
        '-p', '--paramstune',
        nargs='?',
        const=1,
        type=int,
        help='use hp tuner for model_generation'
    )
    parser.add_argument(
        '-a', '--avgplot',
        nargs='?',
        const='',
        type=str,
        help='Iterate through given experiment parent directory and create avg plots in each experiment'
    )
    args = vars(parser.parse_args())

    if args['paramstune']:
        processes = []
        start = time.time()
        model_batch_size = 5
        num_models = list(range(int(args['paramstune'])))
        model_batches = np.array_split(
            num_models, np.arange(model_batch_size, len(num_models), model_batch_size))
        color_selector = ColorSelector()
        for batch in model_batches:
            for model_index in batch:
                control_center = ControlCenter(
                    cc_id=model_index, color=color_selector.get_color(), subdir=subdir)
                proc = Process(target=control_center.train_hp_model)
                time.sleep(1)
                proc.start()
                processes.append(proc)
            for p in processes:
                p.join()
        end = time.time()
        print(f"Tuning took {end - start} seconds.")

    if args['train']:
        control_center = ControlCenter(
            subdir=subdir)
        control_center.train_model()
        if args['render']:
            control_center.render_model(
                control_center.experiment.experiment_dir)
            args['render'] = False

    if args['render']:
        control_center = ControlCenter(subdir=subdir)
        control_center.render_model(args['render'])

    if args['avgplot']:
        control_center = ControlCenter(subdir=subdir)
        control_center.creates_avg_plots(args['avgplot'])
