import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import ndimage
import imageio
from postprocessing.replayer import Replayer
from utils.config_reader import get_experiment_config
from utils.plotting import plot_learning_curve, plot_stacked_area
import math
import os
import matplotlib.pyplot as plt


class BoatEnvironmentRenderer:
    """
    This class is used to visualize the boat enironment that has already been trained. The visualizations are generated by using the replayer to replay previously recorded episodes.
    """

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.config = get_experiment_config(
            experiment_dir, 'tuned_configs.yaml')
        self.image_buffer = []
        self.replayer = Replayer(experiment_dir)

        plot_learning_curve(
            self.replayer.info_data.loc[:, 'episode_reward'].to_numpy(),
            self.config.base_settings.avg_lookback,
            os.path.join(self.experiment_dir, 'plots', 'reward.png'))
        plt.close('all')
        plot_stacked_area(self.experiment_dir)
        plt.close('all')
        self.fig, (self.axb, self.axwa, self.axwf) = plt.subplots(
            3, 1, height_ratios=[3, 1, 1])
        self.plt_objects = {}

        self.current_path = [[], []]
        self.previous_best_index = -1
        self.previous_best_path = [[], []]

        self.episode_index_memory = None

    def create_base_image(self):
        """
        Creates the base image that is being generated for every important episode. This method should be called only one time once the renderer is being constructed.
        """
        xmin = -5
        xmax = self.config.boat_env.goal_line + \
            int(self.config.boat_env.goal_line * 0.1)

        ymin = -self.config.boat_env.track_width - \
            int(self.config.boat_env.track_width * 0.1)
        ymax = self.config.boat_env.track_width + \
            int(self.config.boat_env.track_width * 0.1)
        # Base image axis settings
        self.axb.set_xlim([xmin, xmax])
        self.axb.set_ylim([ymin, ymax])
        self.axb.axhline(self.config.boat_env.track_width, color='black')
        self.axb.axhline(-self.config.boat_env.track_width, color='black')
        self.axb.axvline(self.config.boat_env.goal_line, color='green')

        self.gradient_image(
            self.axb,
            direction=0,
            extent=(xmin, xmax, ymin, 0),
            cmap=plt.cm.Blues,
            cmap_range=(0.1, 0.5)
        )
        self.gradient_image(
            self.axb,
            direction=0,
            extent=(xmin, xmax, 0, ymax),
            cmap=plt.cm.Blues,
            cmap_range=(0.5, 0.1)
        )
        self.axb.set_aspect('auto')

        # Wind image settings and plots
        env_data = self.replayer.episode_data
        wind_xmax = self.replayer.total_dt
        self.axwf.set_xlim([0, wind_xmax])
        self.axwf.set_ylim([0, float(self.config.wind.max_velocity) + 0.01])
        self.axwf.set_yticks([0, float(self.config.wind.max_velocity)])
        self.axwf.set_yticklabels(
            [0, float(self.config.wind.max_velocity)])
        self.axwf.plot(np.arange(0, wind_xmax),
                       env_data.loc[:, 'wind_velocity'].head(wind_xmax),
                       color='blue')

        self.axwa.set_xlim([0, wind_xmax])
        self.axwa.set_ylim([0, np.pi * 2 + 0.01])
        self.axwa.set_yticks([0, np.pi, np.pi * 2])
        self.axwa.set_yticklabels([0, 'π', '2π'])
        self.axwa.get_xaxis().set_visible(False)
        self.axwa.plot(np.arange(0, wind_xmax),
                       env_data.loc[:, 'wind_angle'].head(wind_xmax),
                       color='blue')

    def update_objects_on_image(self, episode_index, dt):
        """
        Creates all objects that get updated with dt/episode index.
        """
        self.create_base_image()
        env_data = self.replayer.episode_data

        self.expand_current_path(env_data, dt)
        self.set_previous_best_path(env_data)

        self.axb.grid(False)
        self.axwa.grid(False)
        self.axwf.grid(False)

        self.plt_objects['text1'] = self.axb.text(
            y=self.config.boat_env.track_width + self.config.boat_env.track_width * 0.1,
            x=-5,
            s=f"dt: {dt}, "
            f"ship angle: {math.degrees(env_data.iloc[dt]['boat_angle']):.2f}, "
            f"actor action: {env_data.iloc[dt]['action_rudder']:.2f}, ",
            fontsize=7
        )

        self.plt_objects['text2'] = self.axb.text(
            y=self.config.boat_env.track_width + self.config.boat_env.track_width * 0.2,
            x=-5,
            s=f"reward: {env_data.iloc[dt]['reward']:.2f}, ",
            fontsize=7
        )

        self.plt_objects['previous_best_path'] = self.axb.plot(
            self.previous_best_path[0],
            self.previous_best_path[1],
            color='orange'
        )

        self.plt_objects['current_path'] = self.axb.plot(
            self.current_path[0],
            self.current_path[1],
            color='red'
        )

        self.plt_objects['boat'] = self.axb.add_artist(
            self.CustomPltObject('boat_topdown.png').set_object_state(
                angle=env_data.iloc[dt]['boat_angle'],
                position=[
                    env_data.iloc[dt]['boat_position_x'],
                    env_data.iloc[dt]['boat_position_y']
                ],
                zoom=1
            )
        )

        # Wind arrow length calculation

        self.plt_objects['velocity_vector'] = self.draw_arrow(
            angle=env_data.iloc[dt]['boat_angle'],
            position=[
                env_data.iloc[dt]['boat_position_x'],
                env_data.iloc[dt]['boat_position_y']
            ],
            length=7,
            color='red'
        )

        # self.plt_objects['action_vector'] = self.draw_arrow(
        #     angle=((np.pi/2) * math.copysign(1, env_data.iloc[dt]['action'])),
        #     position=[
        #         env_data.iloc[dt]['boat_position_x'],
        #         env_data.iloc[dt]['boat_position_y']
        #     ],
        #     length=2,
        #     color='green'
        # )

        self.plt_objects['wind_vector'] = self.draw_arrow(
            angle=env_data.iloc[dt]['wind_angle'],
            position=[
                env_data.iloc[dt]['boat_position_x'],
                env_data.iloc[dt]['boat_position_y']
            ],
            length=2,
            color='blue'
        )

        self.plt_objects['wind_angle_line_indicator'] = self.axwa.axvline(
            dt, color='black'
        )

        # Calculate y pos for text
        wa_text_y = 2 * np.pi * 0.08
        if env_data.iloc[dt]['wind_angle'] < np.pi:
            wa_text_y = 2 * np.pi * 0.85
        self.plt_objects['wind_angle_line_indicator_text'] = self.axwa.text(
            y=wa_text_y,
            x=dt + 2,
            s=f"w_angle {env_data.iloc[dt]['wind_angle']:.2f}",
            fontsize=7
        )

        self.plt_objects['wind_velocity_line_indicator'] = self.axwf.axvline(
            dt, color='black'
        )

        # Calculate y pos for text
        wf_text_y = self.config.wind.max_velocity * 0.08
        if env_data.iloc[dt]['wind_velocity'] < self.config.wind.max_velocity/2:
            wf_text_y = self.config.wind.max_velocity * 0.85
        self.plt_objects['wind_velocity_line_indicator_text'] = self.axwf.text(
            y=wf_text_y,
            x=dt + 2,
            s=f"w_velocity {env_data.iloc[dt]['wind_velocity']:.2f}",
            fontsize=7
        )

    def reset_renderer(self):
        self.image_buffer = []
        self.current_path = [[], []]

    def create_gif_from_buffer(self, episode_index):
        rendering_dir = os.path.join(
            self.experiment_dir, "rendering", f"episode_{episode_index}" + ".gif")
        with imageio.get_writer(rendering_dir, mode="I") as writer:
            for image in self.image_buffer:
                writer.append_data(image)

    def draw_image_to_buffer(self):
        self.fig.canvas.draw()
        self.axb.clear()
        self.axwa.clear()
        self.axwf.clear()
        plt.close("all")
        image_data = np.frombuffer(
            self.fig.canvas.tostring_rgb(),
            dtype=np.uint8
        )
        image_data = image_data.reshape(
            self.fig.canvas.get_width_height()[::-1] + (3,)
        )
        self.image_buffer.append(image_data)

    def gradient_image(self, ax, extent, direction=0.3, cmap_range=(0, 1), **kwargs):
        """
        https://matplotlib.org/3.2.0/gallery/lines_bars_and_markers/gradient_bar.html
        Draw a gradient image based on a colormap.
        """
        phi = direction * np.pi / 2
        v = np.array([np.cos(phi), np.sin(phi)])
        X = np.array([[v @ [1, 0], v @ [1, 1]],
                      [v @ [0, 0], v @ [0, 1]]])
        a, b = cmap_range
        X = a + (b - a) / X.max() * X
        im = ax.imshow(X, extent=extent, interpolation='bicubic',
                       vmin=0, vmax=1, **kwargs)
        return im

    def draw_arrow(self, angle, position, length, color):
        dx = math.cos(angle) * length + position[0]
        dy = math.sin(angle) * length + position[1]
        prop = dict(arrowstyle="->, head_width=0.4, head_length=0.8",
                    color=color, shrinkA=0, shrinkB=0)
        return self.axb.annotate("", xy=(dx, dy), xytext=(
            position[0],
            position[1]),
            arrowprops=prop
        )

    def expand_current_path(self, env_data, dt):
        x, y = (env_data.iloc[dt]['boat_position_x'],
                env_data.iloc[dt]['boat_position_y'])
        self.current_path[0].append(x)
        self.current_path[1].append(y)

    def set_previous_best_path(self, env_data):
        if self.previous_best_tmp != -1:
            self.previous_best_path = [env_data.iloc[:]['boat_position_x'],
                                       env_data.iloc[:]['boat_position_y']]

    class CustomPltObject:
        """
        Creates a custom asset/image that can be added on plots.
        """

        def __init__(self, asset_file):
            self.asset_file = asset_file

        def set_object_state(self, angle, position, zoom):
            base_image = self.load_object_asset()
            rotated_image = ndimage.rotate(
                (base_image * 255).astype(np.uint8),
                np.degrees(angle) - 90
            )
            imagebox = OffsetImage(rotated_image, zoom=zoom)
            annotationbbox = AnnotationBbox(
                imagebox,
                (position[0], position[1]),
                frameon=False
            )
            return annotationbbox

        def load_object_asset(self):
            return mpimg.imread(os.path.join("rendering", "assets", self.asset_file))
