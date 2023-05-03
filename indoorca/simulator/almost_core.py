from typing import List, NamedTuple, Tuple

import numpy as np

from .orca import IndoorORCASim, IndoorORCASimConfig
from indoorca.environment.core import Environment
import indoorca

class Position():
    def __init__(self, map_shape: Tuple[int, int]):
        self.x = 0
        self.y = 0
        self.x_pix = 0
        self.y_pix = 0
        self.x_pix_lims = (0, map_shape[1]-1)
        self.y_pix_lims = (0, map_shape[0]-1)
        self.half_map_width = map_shape[1] / 2
        self.x_lims = (-self.half_map_width / indoorca.pix_per_meter, 
                        self.half_map_width / indoorca.pix_per_meter)
        self.half_map_height = map_shape[0] / 2
        self.y_lims = (-self.half_map_height / indoorca.pix_per_meter,
                        self.half_map_height / indoorca.pix_per_meter)

    def set_position(self, x: float, y: float) -> None:
        #Make sure the position is within the map
        if x < self.x_lims[0]:
            raise ValueError('x position is less than the minimum x position of the map.')
        elif x > self.x_lims[1]:
            raise ValueError('x position is greater than the maximum x position of the map.')
        elif y < self.y_lims[0]:
            raise ValueError('y position is less than the minimum y position of the map.')
        elif y > self.y_lims[1]:
            raise ValueError('y position is greater than the maximum y position of the map.')
            
        self.x = x
        self.y = y
        self.x_pix, self.y_pix = self.meters_to_pix(x, y)

    def set_position_pix(self, x_pix: int, y_pix: int) -> None:
        #Make sure the position is within the map
        if x_pix < self.x_pix_lims[0]:
            raise ValueError('x position is less than the minimum x position of the map.')
        elif x_pix > self.x_pix_lims[1]:
            raise ValueError('x position is greater than the maximum x position of the map.')
        elif y_pix < self.y_pix_lims[0]:
            raise ValueError('y position is less than the minimum y position of the map.')
        elif y_pix > self.y_pix_lims[1]:
            raise ValueError('y position is greater than the maximum y position of the map.')
        
        self.x_pix = x_pix
        self.y_pix = y_pix
        self.x, self.y = self.pix_to_meters(x_pix, y_pix)

    def meters_to_pix(self, x: float, y: float) -> Tuple[int, int]:

        x_pix = int((x+self.half_map_width) * indoorca.pix_per_meter)
        y_pix = int((y+self.half_map_height) * indoorca.pix_per_meter)

        #Make sure the position is within the map
        if x_pix < self.x_pix_lims[0]:
            raise ValueError('x position is less than the minimum x position of the map.')
        elif x_pix > self.x_pix_lims[1]:
            raise ValueError('x position is greater than the maximum x position of the map.')
        elif y_pix < self.y_pix_lims[0]:
            raise ValueError('y position is less than the minimum y position of the map.')
        elif y_pix > self.y_pix_lims[1]:
            raise ValueError('y position is greater than the maximum y position of the map.')
        
        return (x_pix, y_pix)
    
    def pix_to_meters(self, x_pix: int, y_pix: int) -> Tuple[float, float]:
        x = (x_pix / indoorca.pix_per_meter) - self.half_map_width
        y = (y_pix / indoorca.pix_per_meter) - self.half_map_height

        #Make sure the position is within the map
        if x < self.x_lims[0]:
            raise ValueError('x position is less than the minimum x position of the map.')
        elif x > self.x_lims[1]:
            raise ValueError('x position is greater than the maximum x position of the map.')
        elif y < self.y_lims[0]:
            raise ValueError('y position is less than the minimum y position of the map.')
        elif y > self.y_lims[1]:
            raise ValueError('y position is greater than the maximum y position of the map.')
        
        return (x, y)

class Agent():


    def __init__(self):
        """
        # super(Agent, self).__init__()
        # self.collision_filename = os.path.join(
        #     gibson2.assets_path, 'models', 'person_meshes',
        #     'person_{}'.format(style), 'meshes', 'person_vhacd.obj')
        # self.visual_filename = os.path.join(
        #     gibson2.assets_path, 'models', 'person_meshes',
        #     'person_{}'.format(style), 'meshes', 'person.obj')
        # self.visual_only = visual_only
        # self.scale = scale
        """
        self.default_pos = [0,0]
        self.default_orn_euler = np.array([np.pi / 2.0, 0.0, np.pi / 2.0])
        self.yaw = self.default_orn_euler
        self.pos = self.default_pos
        self.orientation = self.default_orn_euler
        self.id = 0
        self.start_position = self.default_pos
        self.goal_position = self.default_pos

    def set_position(self, pos):
        self.pos = pos

    def get_position(self):
        return self.pos

    def _load(self):
        """
        Load the object into pybullet
        """
        pass


    def set_yaw(self, yaw):
        self.yaw = yaw
        self.orientation = [self.orientation[0],
                       self.orientation[1],
                       self.orientation[2] + yaw]
        
        #TODO: add yaw/oritenation and position to the agen
        # euler_angle = [self.default_orn_euler[0],
        #                self.default_orn_euler[1],
        #                self.default_orn_euler[2] + yaw]
        # pos, _ = p.getBasePositionAndOrientation(self.body_id)
        # p.resetBasePositionAndOrientation(
        #     self.body_id, pos, p.getQuaternionFromEuler(euler_angle)
        # )

    def get_yaw(self):
        quat_orientation = self.orientation

        # Euler angles in radians ( roll, pitch, yaw )
        euler_orientation = quat_orientation #p.getEulerFromQuaternion(quat_orientation)

        yaw = euler_orientation[2] - self.default_orn_euler[2]

        return yaw

class MultiAgentSim:
    """
    Multi-agent navigation planning using IndoorORCASimulator. Generates global plans, 
    waypoints, and local plans for each agent.
    """
    def __init__(self, name: str, map: np.ndarray, num_agents: int = 1) -> None:
        self.sim = IndoorORCASim(IndoorORCASimConfig())
        self.env = Environment(name, map)
        self.main_agent_id = 0
        self.agents = {}
        self.ids = []
        self.agent_waypoints = {}
        self.agent_trajectories = {}
        self.obstacles = []
        self.num_agents = num_agents
        self.num_steps_stop = [0] * self.num_agents
        self.neighbor_stop_radius =  1.0
        # By default, stop 2 seconds if stuck
        self.num_steps_stop_thresh = 20
        # backoff when angle is greater than 135 degrees
        self.backoff_radian_thresh = np.deg2rad(135.0)
        self.pedestrian_goal_thresh = 0.3
        self.personal_space_violation_steps = 0


        if num_agents < 1:
            raise ValueError('Number of agents must be greater than or equal to 1.')
        

        #TODO: Add flag to optionally have pedestrians stop rather than back up


        """
        Parameters for our mechanism of preventing pedestrians to back up.
        Instead, stop them and then re-sample their goals.

        num_steps_stop         A list of number of consecutive timesteps
                               each pedestrian had to stop for.
        num_steps_stop_thresh  The maximum number of consecutive timesteps
                               the pedestrian should stop for before sampling
                               a new waypoint.
        neighbor_stop_radius   Maximum distance to be considered a nearby
                               a new waypoint.
        backoff_radian_thresh  If the angle (in radian) between the pedestrian's
                               orientation and the next direction of the next
                               goal is greater than the backoffRadianThresh,
                               then the pedestrian is considered backing off.
        """

        self.num_steps_stop = [0] * self.num_agents
        self.neighbor_stop_radius =  1.0
        # By default, stop 2 seconds if stuck
        self.num_steps_stop_thresh = 20
        # backoff when angle is greater than 135 degrees
        self.backoff_radian_thresh = np.deg2rad(135.0)


        # Threshold of pedestrians reaching the next waypoint
        self.pedestrian_goal_thresh = 0.3
        self.personal_space_violation_steps = 0

        self.load_agents()
        self.agents = self.agents

        #TODO: Load obstacles to Simulator
        self.load_obstacles()

        self.offline_eval = False 
        
        #self.config.get(
          #  'load_scene_episode_config', False)

    def load_obstacles(self) -> None:
        self.env.process_obstacles()
        obs_list = self.env.get_obstacle_meters()
        self.obstacles = self.sim.add_obstacles(obs_list)
        self.sim.process_obstacles()

    def load_agents(self) -> None:

        for _ in range(self.num_agents):
            agent = Agent()
            agent.id = self.sim.add_agent(agent.default_pos)
            self.agents[agent.id] = agent
            self.ids += [agent.id]

    def sample_pos(self) -> List[float]:
        """Sample a random position that is available in the environment

        Returns
        -------
        List[float]
            An x,y position in world coordinates(meters)
        """
        pos=[]
        resample = True
        while resample:
            pos = self.env.get_random_point()
            resample = False
            for agent in self.agents.values():
                dist = np.linalg.norm(pos - agent.get_position())
                if dist < self.sim.config.radius*2:
                    resample = True
                    break   

        return pos


    def sample_initial_pos(self, agent_id:int) -> Tuple[float]:#env, ped_id):
        """
        Sample a new initial position for pedestrian with ped_id.
        The inital position is sampled randomly until the position is
        at least |self.orca_radius| away from all other pedestrians' initial
        positions and the robot's initial position.
        """

        if agent_id not in self.ids:
            raise ValueError("Agent with id {} is not in the simulation.".format(agent_id))
        
        # resample pedestrian's initial position

        #resample = True
        #TODO: Add a max number of resamples
        #TODO: Add check for if the agent is too close to the robots

        # while resample:

        initial_pos = self.env.get_random_point()

        #     # print('initial_pos', initial_pos)
        #     resample = False

        #     # # If too close to the robot, resample
        #     # dist = np.linalg.norm(initial_pos[:2] - self.initial_pos[:2])
        #     # if dist < self.sim.config.radius:
        #     #     resample = True
        #     #     continue

        #     agent = self.agents[agent_id]

        #     for neighbor_id in self.ids:
        #         if neighbor_id == agent_id:
        #             continue
        #         neighbor = self.agents[neighbor_id]
        #         neighbor_pos = neighbor.get_position()

        #         dist = np.linalg.norm(np.array(agent.get_position()) -
        #                               np.array(neighbor_pos))

        #         if dist < self.sim.get_radius()*2:
        #             resample = True
        #             break

        return initial_pos

    def reset_pedestrians(self) -> None:
        """
        Reset the poses of pedestrians to have no collisions with the scene or the robot and set waypoints to follow

        :param env: environment instance
        """
        self.agent_waypoints = {}

        for agent_id, agent in self.agents.items():

            #TODO: Get initial position from graph
            initial_pos = self.sample_initial_pos(agent_id)
            # initial_orn = np.array([np.pi / 2.0, 0.0, np.pi / 2.0])
            # initial_orn = p.getQuaternionFromEuler(ped.default_orn_euler)

            waypoints = self.sample_new_target_pos(initial_pos)
            # print(waypoints)

            # ped.set_position_orientation(initial_pos, initial_orn)
            self.sim.set_agent_position(agent_id, initial_pos)
            self.agent_waypoints[agent_id] = waypoints

    def reset_agent(self)->None: #, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.

        :param env: environment instance
        """
        # super(SocialNavRandomTask, self).reset_agent(env)
        # if self.offline_eval:
        #     self.episode_config.reset_episode()
        #     episode_index = self.episode_config.episode_index
        #     initial_pos = np.array(
        #         self.episode_config.episodes[episode_index]['initial_pos'])
        #     initial_orn = np.array(
        #         self.episode_config.episodes[episode_index]['initial_orn'])
        #     target_pos = np.array(
        #         self.episode_config.episodes[episode_index]['target_pos'])
        #     self.initial_pos = initial_pos
        #     self.target_pos = target_pos
        #     env.robots[0].set_position_orientation(initial_pos, initial_orn)

        # self.sim.setAgentPosition(self.robot_orca_ped,
        #                                tuple(self.initial_pos[0:2]))
        # self.reset_pedestrians(env)
        

        self.personal_space_violation_steps = 0
        self.sim.trajectories = []
        self.sim = IndoorORCASim(IndoorORCASimConfig())
        self.sim.add_obstacles(self.env.get_obstacle_meters())
        self.sim.process_obstacles()
        self.agents = {}
        self.ids = []
        self.load_agents()   
        self.reset_pedestrians()


    def sample_new_target_pos(self, initial_pos):
        """
        Samples a new target position for a pedestrian.
        The target position is read from the saved data for a particular
        pedestrian when |self.offline_eval| is True.
        If False, the target position is sampled from the floor map

        :param env: an environment instance
        :param initial_pos: the pedestrian's initial position
        :param ped_id: the pedestrian id to sample goal
        :return waypoints: the path to the goal position
        """

        while True:

            #TODO: Sample or get target position from graph
            #Sample until target position is at least 4 meters away
            furthest_pos = None
            furthest_dist = 0.
            for _ in range(1000):
                target_pos = self.env.get_random_point()
                dist = np.linalg.norm(np.array(initial_pos) - np.array(target_pos))
                if dist > furthest_dist:
                    furthest_pos = target_pos
                    furthest_dist = dist

                if dist > 5.:
                    furthest_pos = target_pos
                    break
            
            target_pos = furthest_pos
            # target_pos = self.env.find_goal_at_avg_shortest_path_length(initial_pos)
            # print(target_pos)
            shortest_path, _ = self.env.shortest_path(np.array(initial_pos), np.array(target_pos))
            print('PATH')
            print(shortest_path)
            #self.get_shortest_path(initial_pos, target_pos)
            if len(shortest_path) > 1:
                break

        waypoints = self.shortest_path_to_waypoints(shortest_path)
        print('WAYPOINTS')
        print(waypoints)
        print('#'*20)


        return waypoints

    def shortest_path_to_waypoints(self, shortest_path):
        # Convert dense waypoints of the shortest path to coarse waypoints
        # in which the collinear waypoints are merged.
        # assert len(shortest_path) > 0
        # waypoints = []
        # valid_waypoint = None
        # prev_waypoint = None
        # cached_slope = None
        # for waypoint in shortest_path:
        #     if valid_waypoint is None:
        #         valid_waypoint = waypoint
        #     elif cached_slope is None:
        #         cached_slope = waypoint - valid_waypoint
        #     else:
        #         cur_slope = waypoint - prev_waypoint
        #         cosine_angle = np.dot(cached_slope, cur_slope) / \
        #             (np.linalg.norm(cached_slope) * np.linalg.norm(cur_slope))
        #         if np.abs(cosine_angle - 1.0) > 1e-3:
        #             waypoints.append(valid_waypoint)
        #             valid_waypoint = prev_waypoint
        #             cached_slope = waypoint - valid_waypoint

        #     prev_waypoint = waypoint

        # # Add the last two valid waypoints
        # waypoints.append(valid_waypoint)
        # waypoints.append(shortest_path[-1])

        # # Remove the first waypoint because it's the same as the initial pos
        # waypoints.pop(0)

        # return waypoints
    
            # Convert dense waypoints of the shortest path to coarse waypoints
        # in which the collinear waypoints are merged.
        assert len(shortest_path) > 0
        waypoints = []
        valid_waypoint = None
        prev_waypoint = None
        cached_slope = None
        for waypoint in shortest_path:
            if valid_waypoint is None:
                valid_waypoint = waypoint
            elif cached_slope is None:
                cached_slope = waypoint - valid_waypoint
            else:
                cur_slope = waypoint - prev_waypoint
                cosine_angle = np.dot(cached_slope, cur_slope) / \
                    (np.linalg.norm(cached_slope) * np.linalg.norm(cur_slope))
                if np.abs(cosine_angle - 1.0) > 1e-3:
                    waypoints.append(valid_waypoint)
                    valid_waypoint = prev_waypoint
                    cached_slope = waypoint - valid_waypoint

            prev_waypoint = waypoint

        # Add the last two valid waypoints
        waypoints.append(valid_waypoint)
        waypoints.append(shortest_path[-1])

        # Remove the first waypoint because it's the same as the initial pos
        waypoints.pop(0)

        return waypoints
    

    def step(self):
        """
        Perform task-specific step: move the pedestrians based on ORCA while
        disallowing backing up

        :param env: environment instance
        """

        #TODO: Remove self.pedestrians from everything and instead get everything from simulator
            
        for agent_id in self.ids:
            agent = self.agents[agent_id]
            agent_pos = agent.get_position()
            waypoints = self.agent_waypoints[agent_id]

            # if len(waypoints) == 0:
            #     raise ValueError("The waypoints for agent {} is empty".format(
            #         agent_id))
            
            current_pos = np.array(agent_pos)

            # Sample new waypoints if empty OR
            # if the pedestrian has stopped for self.num_steps_stop_thresh steps
            if len(waypoints) == 0 or \
                    self.num_steps_stop[agent_id] >= self.num_steps_stop_thresh:
                # if self.offline_eval:
                #     waypoints = self.sample_new_target_pos(env, current_pos, i)
                # else:
                print("Sampling new target position for agent {}".format(agent_id))

                waypoints = self.sample_new_target_pos(current_pos)
                self.agent_waypoints[agent_id] = waypoints
                self.num_steps_stop[agent_id] = 0

            # print("Agent {} has {} waypoints".format(agent_id, len(waypoints)))
            print('Agent {} waypoints: {}'.format(agent_id, waypoints))
            next_goal = waypoints[0]
            # self.pedestrian_goals[i].set_position(
            #     np.array([next_goal[0], next_goal[1], current_pos[2]]))

            yaw = np.arctan2(next_goal[1] - current_pos[1],
                             next_goal[0] - current_pos[0])
            agent.set_yaw(yaw)
            
            desired_vel = next_goal - current_pos
            desired_vel = desired_vel / \
                np.linalg.norm(desired_vel) * self.sim.get_max_speed()
            self.sim.set_agent_pref_velocity(agent_id, list(desired_vel))

        self.sim.do_step()

        # # Update the pedestrian position in PyBullet if it does not stop

        next_peds_pos_xyz, next_peds_stop_flag = \
            self.update_pos_and_stop_flags()

        # Update the pedestrian position in PyBullet if it does not stop
        # Otherwise, revert back the position in RVO2 simulator
            
        for agent_id in self.ids:
            agent = self.agents[agent_id]
            waypoints = self.agent_waypoints[agent_id]
            pos_xyz = next_peds_pos_xyz[agent_id]

            if next_peds_stop_flag[agent_id] is True:
                # revert back ORCA sim pedestrian to the previous time step
                self.num_steps_stop[agent_id] += 1
                self.sim.set_agent_position(agent_id, pos_xyz[:2])
            else:
                # advance pybullet pedstrian to the current time step
                self.num_steps_stop[agent_id] = 0
                agent.set_position(pos_xyz)
                next_goal = waypoints[0]
                if np.linalg.norm(next_goal - np.array(pos_xyz[:2])) \
                        <= self.pedestrian_goal_thresh:
                    waypoints.pop(0)

        # Detect robot's personal space violation
        # personal_space_violation = False
        # robot_pos = env.robots[0].get_position()[:2]
        # for agent in self.pedestrians:
        #     ped_pos = agent.get_position()[:2]
        #     if l2_distance(robot_pos, ped_pos) < self.orca_radius:
        #         personal_space_violation = True
        #         break
        # if personal_space_violation:
        #     self.personal_space_violation_steps += 1

    def update_pos_and_stop_flags(self):
        """
        Wrapper function that updates pedestrians' next position and whether
        they should stop for the next time step

        :return: the list of next position for all pedestrians,
                 the list of flags whether the pedestrian should stop for the
                 next time step
        """
        next_peds_pos_xy = \
            {agent_id: self.sim.get_agent_position(agent_id) for agent_id in self.ids}
        next_peds_stop_flag = {agent_id: False for agent_id in self.ids}
    
        for agent_id in self.ids:

            agent = self.agents[agent_id]
            pos_xy = self.sim.get_agent_position(agent_id)
            # prev_pos_xy = np.array(agent.get_position())
            next_pos_xy = np.array([pos_xy[0], pos_xy[1]])


            # prev_pos_xyz = ped.get_position()
            # next_pos_xyz = np.array([pos_xy[0], pos_xy[1], prev_pos_xyz[2]])

            # if self.detect_backoff(agent_id):
            #     self.stop_neighbor_pedestrians(agent_id,
            #                                    next_peds_stop_flag,
            #                                    next_peds_pos_xy)
            # elif next_peds_stop_flag[agent_id] is False:
                # If there are no other neighboring pedestrians that forces
                # this pedestrian to stop, then simply update next position.
            next_peds_pos_xy[agent_id] = next_pos_xy

        return next_peds_pos_xy, next_peds_stop_flag

    def stop_neighbor_pedestrians(self, agent_id, peds_stop_flags, peds_next_pos_xy):
        """
        If the pedestrian whose instance stored in self.pedestrians with
        index |id| is attempting to backoff, all the other neighboring
        pedestrians within |self.neighbor_stop_radius| will stop

        :param id: the index of the pedestrian object
        :param peds_stop_flags: list of boolean corresponding to if the pestrian
                                at index i should stop for the next
        :param peds_next_pos_xyz: list of xyz position that the pedestrian would
                            move in the next timestep or the position in the
                            PyRVOSimulator that the pedestrian would revert to
        """
        agent = self.agents[agent_id]
        ped_pos_xyz = agent.get_position()

        # for i, neighbor in enumerate(self.pedestrians):
        for agent_id in self.ids:
            neighbor = self.agents[agent_id]
            if agent_id == neighbor.id:
                continue
            neighbor_pos_xyz = neighbor.get_position()
            dist = np.linalg.norm([neighbor_pos_xyz[0] - ped_pos_xyz[0],
                                   neighbor_pos_xyz[1] - ped_pos_xyz[1]])
            if dist <= self.neighbor_stop_radius:
                peds_stop_flags[agent_id] = True
                peds_next_pos_xy[agent_id] = neighbor_pos_xyz
        peds_stop_flags[agent_id] = True
        peds_next_pos_xy[agent_id] = ped_pos_xyz

    def detect_backoff(self, agent_id):
        """
        Detects if the pedestrian is attempting to perform a backoff
        due to some form of imminent collision

        :param ped: the pedestrain object
        :param orca_ped: the pedestrian id in the orca simulator
        :return: whether the pedestrian is backing off
        """
        agent = self.agents[agent_id]
        pos_xy = self.sim.get_agent_position(agent_id)
        prev_pos_xyz = agent.get_position()

        yaw = agent.get_yaw()

        # Computing the directional vectors from yaw
        normalized_dir = np.array([np.cos(yaw), np.sin(yaw)])

        next_dir = np.array([pos_xy[0] - prev_pos_xyz[0],
                             pos_xy[1] - prev_pos_xyz[1]])

        if np.linalg.norm(next_dir) == 0.0:
            return False

        next_normalized_dir = next_dir / np.linalg.norm(next_dir)

        angle = np.arccos(np.dot(normalized_dir, next_normalized_dir))
        return angle >= self.backoff_radian_thresh
    
    #TODO: Finish termination condition

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """

        #TODO: Get done and info flags from simulation

        # done, info = super(SocialNavRandomTask, self).get_termination(
        #     env, collision_links, action, info)
        # done = True

        raise NotImplementedError

        # if done:
        #     info['psc'] = 1.0 - (self.personal_space_violation_steps /
        #                          env.config.get('max_step', 500))
        #     if self.offline_eval:
        #         episode_index = self.episode_config.episode_index
        #         orca_timesteps = self.episode_config.episodes[episode_index]['orca_timesteps']
        #         info['stl'] = float(info['success']) * \
        #             min(1.0, orca_timesteps / env.current_step)
        #     else:
        #         info['stl'] = float(info['success'])
        # else:
        #     info['psc'] = 0.0
        #     info['stl'] = 0.0

        # return done, info