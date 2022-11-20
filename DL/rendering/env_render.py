import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import ndimage
import imageio

class EnvironmentRenderer:
    """
    A visulization tool that uses the gym render method to render the environment to the screen in real time by using matplotlib.
    """
    def __init__(self, config, title=None):
        self.config = config
        self.image_buffer = []
        self.current_path = [[], []]
        self.previous_best_path = [[], []]
        self.title = title

    def reset_renderer(self):
        self.image_buffer = []
        self.current_path = [[], []]

    def create_gif_from_buffer(self, experiment_path, file_name):
        with imageio.get_writer(os.path.join(experiment_path, file_name + ".gif"), mode="I") as writer:
            for image in self.image_buffer:
                writer.append_data(image)

    def create_new_image(self, env):
        """
        Creates a new image of the current state of the env that is being passed.
        """
        # Save this position
        self.current_path[0].append(env.boat.position[0])
        self.current_path[1].append(env.boat.position[1])

        # Create base plot without boat
        fig, ax = plt.subplots()
        xmin, xmax = -5, self.config.boat_env.boat_fuel + 10
        ymin, ymax = -self.config.boat_env.track_width - self.config.boat_env.track_width_offset, self.config.boat_env.track_width + self.config.boat_env.track_width_offset
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.axhline(self.config.boat_env.track_width, color="black")
        ax.axhline(-self.config.boat_env.track_width, color="black")
        ax.axvline(self.config.boat_env.goal_line, color="green")
        ax.set_title(self.title)

        ax.text(
            0,
            self.config.boat_env.track_width + self.config.boat_env.track_width_offset + 0.1,
            f"Current step: {env.boat.current_step}, "
            f"steps until wind change: {env.boat.steps_until_wind_change}, "
            f"ship angle: {math.degrees(env.boat.angle):.2f}, "
            f"actor action: {env.action[0]:.2f}, "
            f"wind action: {(env.boat.current_wind_angle * env.boat.current_wind_force):.2f}",
            fontsize=7
        )

        # Plot current path and previous best path
        ax.plot(self.previous_best_path[0], self.previous_best_path[1], color="orange")
        ax.plot(self.current_path[0], self.current_path[1], color="red")

        self.gradient_image(
            ax,
            direction=0,
            extent=(xmin, xmax, ymin, 0),
            cmap=plt.cm.Blues,
            cmap_range=(0.1, 0.5)
        )
        self.gradient_image(
            ax,
            direction=0,
            extent=(xmin, xmax, 0, ymax),
            cmap=plt.cm.Blues,
            cmap_range=(0.5, 0.1)
        )
        ax.set_aspect('auto')

        # Render boat
        boat_imagebox = self.render_asset(env.boat.angle, env.boat.position, 1, "boat_topdown.png")
        ax.add_artist(boat_imagebox)

        # Draw boat velocity arrow
        self.draw_arrow(ax, env.boat.angle, env.boat.position, 7, "red")

        # Draw action arrow
        self.draw_arrow(
            ax,
            math.radians(90 * math.copysign(1, env.action[0])),
            env.boat.position,
            2 * np.abs(env.action[0]),
            "green"
        )

        # Draw current wind direction & force
        if env.boat.current_wind_angle != 0:
            self.draw_arrow(
                ax,
                math.radians(90 * env.boat.current_wind_angle),
                env.boat.position,
                2 * env.boat.current_wind_angle,
                "blue"
            )

        # Create image and save to image_buffer
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.image_buffer.append(image_data)
        plt.close("all")

    def render_asset(self, angle, position, zoom, asset):
        """
        Renders the boat with it's current angle
        """
        x, y = position[0], position[1]
        boat_image = self.load_asset(asset)
        rotated_boat_image = ndimage.rotate((boat_image * 255).astype(np.uint8), math.degrees(angle) - 90)
        imagebox = OffsetImage(rotated_boat_image, zoom=zoom)
        imagebox = AnnotationBbox(imagebox, (x, y), frameon=False)
        return imagebox

    def gradient_image(self, ax, extent, direction=0.3, cmap_range=(0, 1), **kwargs):
        """
        https://matplotlib.org/3.2.0/gallery/lines_bars_and_markers/gradient_bar.html
        Draw a gradient image based on a colormap.

        Parameters
        ----------
        ax : Axes
            The axes to draw on.
        extent
            The extent of the image as (xmin, xmax, ymin, ymax).
            By default, this is in Axes coordinates but may be
            changed using the *transform* kwarg.
        direction : float
            The direction of the gradient. This is a number in
            range 0 (=vertical) to 1 (=horizontal).
        cmap_range : float, float
            The fraction (cmin, cmax) of the colormap that should be
            used for the gradient, where the complete colormap is (0, 1).
        **kwargs
            Other parameters are passed on to `.Axes.imshow()`.
            In particular useful is *cmap*.
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

    def draw_arrow(self, ax, angle, position, length, color):
        dx = math.cos(angle) * length + position[0]
        dy = math.sin(angle) * length + position[1]
        prop = dict(arrowstyle="->, head_width=0.4, head_length=0.8", color=color, shrinkA=0, shrinkB=0)
        ax.annotate("", xy=(dx, dy), xytext=(position[0], position[1]), arrowprops=prop)

    def set_best_path(self):
        self.previous_best_path = self.current_path

    def load_asset(self, asset):
        return mpimg.imread(os.path.join("rendering", "assets", asset))