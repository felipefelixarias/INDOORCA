from typing import List, NamedTuple, Tuple
import logging

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib import animation
from numpy.linalg import norm
import cv2

class VisualizerConfig(NamedTuple):
 
    window_size: int = 10
    time_step: float = 0.2
    robot_idx: int = 0
    robot_color: str = 'blue'
    goal_color: str = 'red'
    arrow_color: str = 'red'
    arrow_style: str = '->'
    arrow_head_length: int = 4
    arrow_head_width: int = 2
    cmap: str = 'hsv'
    cmap_num: int = 10
    x_offset: float = 0.11
    y_offset: float = 0.11
    x_lim: List[float] = [-4, 4]
    y_lim: List[float] = [-4, 4]
    x_label: str = 'x(m)'
    y_label: str = 'y(m)'
    x_label_size: int = 16
    y_label_size: int = 16
    legend_font_size: int = 16
    legend_robot: str = 'Robot'
    legend_goal: str = 'Goal'
    legend_pedestrians: str = 'Pedestrians'
    fig_size: List[int] = [7,7]
    time_font_size: int = 8
    pix_per_meter: int = 100
    display_waypoints: bool = True


class Visualizer:
    """Class to visualize the trajectories of the robot and pedestrians in previous episodes. 
    Where the input is a list of pedestrian trajectories and robot trajectory."""

    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.ped_trajectories = None
        self.robot_trajectory = None
        self.map_img = None
        self.num_peds = 0
        self.episode_length = 0
        self.cmap = plt.cm.get_cmap(self.config.cmap, self.config.cmap_num)
        self.waypoints= []

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    def set_trajectories(self, robot_trajectory: List[List[float]], ped_trajectories: List[List[List[float]]]):
        """Set the robot and pedestrian trajectories.

        Parameters
        ----------
        robot_trajectory: 
            The trajectory of the robot.
        ped_trajectories: 
            The trajectories of the pedestrians.
        """
        self.robot_trajectory = robot_trajectory
        self.ped_trajectories = ped_trajectories
        self.num_peds = len(ped_trajectories)
        self.episode_length = len(robot_trajectory)

    def add_map_background(self, map_img: np.ndarray):
        """Add the map background to the visualization.

        Parameters
        ----------
        map_img: 
            The map image.
        """
        self.map_img = map_img

    def add_waypoints(self, waypoints: List[List[float]]):
        """Add the waypoints to the visualization.

        Parameters
        ----------
        waypoints: 
            The waypoints.
        """


        # self.config.display_waypoints = True
        self.waypoints.extend(waypoints)
        #Add the waypoints as points to the plot
        # for waypoint in waypoints:
        #     plt.plot(waypoint[0], waypoint[1], 'o', color='red')



    def render(self, mode: str='traj', output_file: str=None):
        """Render the visualization.

        Parameters
        ----------
        mode: 
            The mode to render the visualization. Defaults to 'traj'.
        output_file: 
        The path to save the video. Defaults to None.
        """
        if mode == 'traj':
            self._render_traj(output_file)
        elif mode == 'video':
            self._render_video(output_file)

    def _render_traj(self, output_file: str=None):
        """Render the trajectory of the robot and pedestrians.
        
        Parameters
        ----------
        output_file: 
            The path to save the video. Defaults to None.
        """

        fig, ax = plt.subplots(figsize=self.config.fig_size)
        ax.tick_params(labelsize=self.config.x_label_size)
        #Add map image to the background
        if self.map_img is not None:
            #Convert the image to RGB format as it is currently [0,1]
            #Turn the image upside down as the origin is at the top left corner
            map_img = np.flipud(self.map_img)
            #Get size of the image in meters
            # print(map_img.shape[0])
            map_size_x = map_img.shape[1] / self.config.pix_per_meter
            map_size_y = map_img.shape[0] / self.config.pix_per_meter
            #Account for the origin being at the center of the image
            map_offset_x = map_size_x / 2
            map_offset_y = map_size_y / 2
            #Set the extent of the image
            extent = [-map_offset_x, map_offset_x, -map_offset_y, map_offset_y]
            map_img = map_img.astype(np.uint8)
            map_img = cv2.cvtColor(map_img*255, cv2.COLOR_GRAY2RGB)

            ax.imshow(map_img, extent=extent)
            self.config.x_lim[0], self.config.x_lim[1] = \
                  -map_offset_x, map_offset_x
            self.config.y_lim[0], self.config.y_lim[1] = \
                    -map_offset_y, map_offset_y

        ax.set_xlim(self.config.x_lim)
        ax.set_ylim(self.config.y_lim)
        ax.set_xlabel(self.config.x_label, fontsize=self.config.x_label_size)
        ax.set_ylabel(self.config.y_label, fontsize=self.config.y_label_size)

        robot_positions = self.robot_trajectory
        ped_positions = self.ped_trajectories

        for k in range(self.episode_length):
            if k % 4 == 0 or k == self.episode_length - 1:
                robot = plt.Circle(robot_positions[k], 0.125, fill=False, color=self.config.robot_color)
                # Check if there are pedestrians in the scene
                if self.num_peds > 0:
                    pedestrians = [plt.Circle(ped_positions[i][k], 0.125, fill=False, color=self.cmap(i)) for i in range(self.num_peds)]
                else:
                    pedestrians = []

                ax.add_artist(robot)
                for ped in pedestrians:
                    ax.add_artist(ped)

            # add time annotation
            global_time = k * self.config.time_step
            if global_time % 4 == 0 or k == self.episode_length - 1:
                agents = pedestrians + [robot]
                #check if there are pedestrians in the scene
                if self.num_peds > 0:
                    times = [plt.text(agents[i].center[0] - self.config.x_offset, agents[i].center[1] - self.config.y_offset,
                                    '{:.1f}'.format(global_time),
                                    color='black', fontsize=self.config.time_font_size) for i in range(len(ped_positions) + 1)]
                else:
                    times = [plt.text(agents[i].center[0] - self.config.x_offset, agents[i].center[1] - self.config.y_offset,
                                    '{:.1f}'.format(global_time),
                                    color='black', fontsize=self.config.time_font_size) for i in range(1)]

                for time in times:
                    ax.add_artist(time)

            if k != 0 and k < self.episode_length-1:
                nav_direction = plt.Line2D((robot_positions[k - 1][0], robot_positions[k][0]),
                                            (robot_positions[k - 1][1], robot_positions[k][1]),
                                            color=self.config.robot_color, ls='solid')
                
                # Check if there are pedestrians in the scene
                if self.num_peds > 0:
                    ped_directions = [plt.Line2D((ped_positions[i][k - 1][0], ped_positions[i][k][0]),
                                                (ped_positions[i][k - 1][1], ped_positions[i][k][1]),
                                                color=self.cmap(i), ls='solid')
                                    for i in range(self.num_peds)]
                else:
                    ped_directions = []

                ax.add_artist(nav_direction)
                for ped_dir in ped_directions:
                    ax.add_artist(ped_dir)

        # add legend
        # robot_legend = mlines.Line2D([], [], color=self.config.robot_color, marker='o', linestyle='None',
        #                             markersize=10, label=self.config.legend_robot)
        # goal_legend = mlines.Line2D([], [], color=self.config.goal_color, marker='o', linestyle='None',
        #                             markersize=10, label=self.config.legend_goal)
        
        if self.config.display_waypoints:
            waypoints = self.waypoints
            for waypoint in waypoints:
                plt.plot(waypoint[0], waypoint[1], 'o', color='red')
        
        
        plt.show()

    def _render_video(self, output_file: str=None):
        """Render the video of the robot and pedestrians.
        
        Parameters
        ----------
        output_file: 
            The path to save the video. Defaults to None.
        """


        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        fig, ax = plt.subplots(figsize=(7,7))
        ax.tick_params(labelsize=self.config.x_label_size)

        #Add map image to the background
        if self.map_img is not None:
            #Convert the image to RGB format as it is currently [0,1]
            #Turn the image upside down as the origin is at the top left corner
            map_img = np.flipud(self.map_img)
            #Get size of the image in meters
            print(map_img.shape[0])
            map_size_x = map_img.shape[1] / self.config.pix_per_meter
            map_size_y = map_img.shape[0] / self.config.pix_per_meter
            #Account for the origin being at the center of the image
            map_offset_x = map_size_x / 2
            map_offset_y = map_size_y / 2
            #Set the extent of the image
            extent = [-map_offset_x, map_offset_x, -map_offset_y, map_offset_y]
            map_img = map_img.astype(np.uint8)
            map_img = cv2.cvtColor(map_img*255, cv2.COLOR_GRAY2RGB)

            ax.imshow(map_img, extent=extent)
            self.config.x_lim[0], self.config.x_lim[1] = \
                  -map_offset_x, map_offset_x
            self.config.y_lim[0], self.config.y_lim[1] = \
                    -map_offset_y, map_offset_y

        ax.set_xlim(self.config.x_lim)
        ax.set_ylim(self.config.y_lim)
        ax.set_xlabel(self.config.x_label, fontsize=self.config.x_label_size)
        ax.set_ylabel(self.config.y_label, fontsize=self.config.y_label_size)


        robot = plt.Circle(self.robot_trajectory[0], 0.125, fill=True, color=robot_color)
        ax.add_artist(robot)

        # Check if there are pedestrians in the scene
        if self.num_peds > 0:
            pedestrians = [plt.Circle(self.ped_trajectories[i][0], 0.125, fill=False, color=cmap(i))
                        for i in range(self.num_peds)]
            for ped in pedestrians:
                ax.add_artist(ped)
        else:
            pedestrians = []

        # add time annotation
        global_time = 0
        agents = pedestrians + [robot]
        #check if there are pedestrians in the scene
        if self.num_peds > 0:
            times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                            '{:.1f}'.format(global_time),
                            color='black', fontsize=self.config.time_font_size) for i in range(self.num_peds + 1)]
        else:
            times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                            '{:.1f}'.format(global_time),
                            color='black', fontsize=self.config.time_font_size) for i in range(1)]

        for time in times:
            ax.add_artist(time)

        radius = 0.15
        
        orientations = []
        for i in range(1 + self.num_peds):
            orientation = []
            if i == 0:
                theta = np.arctan2(self.robot_trajectory[1][1] - self.robot_trajectory[0][1],
                                    self.robot_trajectory[1][0] - self.robot_trajectory[0][0])
                orientation.append(((self.robot_trajectory[0][0], self.robot_trajectory[0][1]),
                                    (self.robot_trajectory[0][0] + radius * np.cos(theta),
                                    self.robot_trajectory[0][1] + radius * np.sin(theta))))
                
            else:
                theta = np.arctan2(self.ped_trajectories[i - 1][1][1] - self.ped_trajectories[i - 1][0][1],
                                    self.ped_trajectories[i - 1][1][0] - self.ped_trajectories[i - 1][0][0])
                orientation.append(((self.ped_trajectories[i - 1][0][0], self.ped_trajectories[i - 1][0][1]),
                                    (self.ped_trajectories[i - 1][0][0] + radius * np.cos(theta),
                                    self.ped_trajectories[i - 1][0][1] + radius * np.sin(theta))))
            orientations.append(orientation)

        arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                    for orientation in orientations]
        
        for arrow in arrows:
            ax.add_artist(arrow)

        global_step = 0

        def update(frame_num: int):
            """Update the animation.
            
            Parameters
            ----------
            frame_num: 
                The current frame number.
            """
            nonlocal global_step
            nonlocal global_time
            global_step += 1
            global_time += self.config.time_step

            # update robot position
            robot.center = self.robot_trajectory[frame_num]

            # update pedestrian positions
            if self.num_peds> 0:
                for i in range(self.num_peds):
                    pedestrians[i].center = self.ped_trajectories[i][frame_num]

            # update time annotation
            for i in range(len(times)):
                times[i].set_position((agents[i].center[0] - x_offset, agents[i].center[1] - y_offset))
                times[i].set_text('{:.1f}'.format(global_time))

            # update robot orientation
            theta = np.arctan2(self.robot_trajectory[frame_num][1] - self.robot_trajectory[frame_num - 1][1],
                                self.robot_trajectory[frame_num][0] - self.robot_trajectory[frame_num - 1][0])
            orientations[0][0] = ((self.robot_trajectory[frame_num][0], self.robot_trajectory[frame_num][1]),
                                    (self.robot_trajectory[frame_num][0] + radius * np.cos(theta),
                                    self.robot_trajectory[frame_num][1] + radius * np.sin(theta)))
            arrows[0].set_positions(*orientations[0][0])

            # update pedestrian orientations
            if self.num_peds > 0:
                for i in range(self.num_peds):
                    theta = np.arctan2(self.ped_trajectories[i][frame_num][1] - self.ped_trajectories[i][frame_num - 1][1],
                                        self.ped_trajectories[i][frame_num][0] - self.ped_trajectories[i][frame_num - 1][0])
                    orientations[i + 1][0] = ((self.ped_trajectories[i][frame_num][0], self.ped_trajectories[i][frame_num][1]),
                                            (self.ped_trajectories[i][frame_num][0] + radius * np.cos(theta),
                                            self.ped_trajectories[i][frame_num][1] + radius * np.sin(theta)))
                    arrows[i + 1].set_positions(*orientations[i + 1][0])

            # return pedestrians + [robot] + arrows + times

        anim = animation.FuncAnimation(fig, update, frames=self.episode_length, interval=500, blit=False)
        anim.running = True

        
        if output_file is not None and output_file[-4] == '.gif':
            pillow_writer = animation.PillowWriter(fps=32)
            anim.save(output_file, writer=pillow_writer)
        elif output_file is not None:
            ffmpeg_writer = animation.writers['ffmpeg']
            writer = ffmpeg_writer(fps=32, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(output_file, writer=writer)
        else:
            plt.show()

    




def render(self, mode='human', output_file=None):

    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    x_offset = 0.11
    y_offset = 0.11
    cmap = plt.cm.get_cmap('hsv', 10)
    # robot_color = 'yellow'
    goal_color = 'red'
    arrow_color = 'red'
    arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

    if mode == 'video':
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.tick_params(labelsize=16)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)

        # add robot and its goal
        robot_positions = [state[0].position for state in self.states]
        goal = mlines.Line2D([0], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=self.config.robot_color)
        ax.add_artist(robot)
        ax.add_artist(goal)
        plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

        # add humans and their numbers
        human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
        humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                    for i in range(len(self.humans))]
        human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                    color='black', fontsize=12) for i in range(len(self.humans))]
        for i, human in enumerate(humans):
            ax.add_artist(human)
            ax.add_artist(human_numbers[i])

        # add time annotation
        time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
        ax.add_artist(time)

        # compute orientation in each step and use arrow to show the direction
        radius = self.robot.radius
        if self.robot.kinematics == 'unicycle':
            orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                            state[0].py + radius * np.sin(state[0].theta))) for state
                            in self.states]
            orientations = [orientation]
        else:
            orientations = []
            for i in range(self.human_num + 1):
                orientation = []
                for state in self.states:
                    if i == 0:
                        agent_state = state[0]
                    else:
                        agent_state = state[1][i - 1]
                    theta = np.arctan2(agent_state.vy, agent_state.vx)
                    orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                            agent_state.py + radius * np.sin(theta))))
                orientations.append(orientation)
        arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                    for orientation in orientations]

        for arrow in arrows:
            ax.add_artist(arrow)
        global_step = 0

        def update(frame_num):
            nonlocal global_step
            nonlocal arrows
            global_step = frame_num
            robot.center = robot_positions[frame_num]
            for i, human in enumerate(humans):
                human.center = human_positions[frame_num][i]
                human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                for arrow in arrows:
                    arrow.remove()
                arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                    arrowstyle=arrow_style) for orientation in orientations]
                for arrow in arrows:
                    ax.add_artist(arrow)

            time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

        def plot_value_heatmap():
            assert self.robot.kinematics == 'holonomic'
            for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                            agent.vx, agent.vy, agent.theta))
            # when any key is pressed draw the action value plot
            fig, axis = plt.subplots()
            speeds = [0] + self.robot.policy.speeds
            rotations = self.robot.policy.rotations + [np.pi * 2]
            r, th = np.meshgrid(speeds, rotations)
            z = np.array(self.action_values[global_step % len(self.states)][1:])
            z = (z - np.min(z)) / (np.max(z) - np.min(z))
            z = np.reshape(z, (16, 5))
            polar = plt.subplot(projection="polar")
            polar.tick_params(labelsize=16)
            mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
            plt.plot(rotations, r, color='k', ls='none')
            plt.grid()
            cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
            cbar = plt.colorbar(mesh, cax=cbaxes)
            cbar.ax.tick_params(labelsize=16)
            plt.show()

        def on_click(event):
            anim.running ^= True
            if anim.running:
                anim.event_source.stop()
                if hasattr(self.robot.policy, 'action_values'):
                    plot_value_heatmap()
            else:
                anim.event_source.start()

        fig.canvas.mpl_connect('key_press_event', on_click)
        anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
        anim.running = True

        if output_file is not None:
            ffmpeg_writer = animation.writers['ffmpeg']
            writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(output_file, writer=writer)
        else:
            plt.show()
    else:
        raise NotImplementedError