from typing import List, NamedTuple, Tuple

import numpy as np

from .orca import IndoorORCASim, IndoorORCASimConfig
from environment.core import Environment
from processing.core import MapProcessor

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

    def set_position(self, pos):
        self.pos = pos

    def get_position(self):
        return self.pos

    def _load(self):
        """
        Load the object into pybullet
        """
        pass
    """
        collision_id = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=self.collision_filename,
            meshScale=[self.scale] * 3)
        visual_id = p.createVisualShape(
            p.GEOM_MESH,
            fileName=self.visual_filename,
            meshScale=[self.scale] * 3)
        if self.visual_only:
            body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visual_id)
        else:
            body_id = p.createMultiBody(baseMass=60,
                                        baseCollisionShapeIndex=collision_id,
                                        baseVisualShapeIndex=visual_id)
        p.resetBasePositionAndOrientation(
            body_id,
            [0.0, 0.0, 0.0],
            p.getQuaternionFromEuler(self.default_orn_euler)
        )
        return body_id

    """

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
        self.map_processor = MapProcessor(map)
        self.agents = {}
        self.ids = []
        self.agent_waypoints = {}
        self.obstacles = []
        self.num_pedestrians = num_agents
        self.num_steps_stop = [0] * self.num_pedestrians
        self.neighbor_stop_radius =  1.0
        # By default, stop 2 seconds if stuck
        self.num_steps_stop_thresh = 20
        # backoff when angle is greater than 135 degrees
        self.backoff_radian_thresh = np.deg2rad(135.0)
        self.pedestrian_goal_thresh = 0.3
        self.personal_space_violation_steps = 0


        if num_agents < 1:
            raise ValueError('Number of agents must be greater than or equal to 1.')
        
        self.num_pedestrians = num_agents
       

        #TODO: Add obstacles to the simulator
        #self.obstacles = self.sim.add_obstacles(self.map_processor.obstacles)

        

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

        self.num_steps_stop = [0] * self.num_pedestrians
        self.neighbor_stop_radius =  1.0
        # By default, stop 2 seconds if stuck
        self.num_steps_stop_thresh = 20
        # backoff when angle is greater than 135 degrees
        self.backoff_radian_thresh = np.deg2rad(135.0)


        # Threshold of pedestrians reaching the next waypoint
        self.pedestrian_goal_thresh = 0.3
        self.personal_space_violation_steps = 0

        self.load_agents()
        self.pedestrians = self.agents



        #TODO: Load obstacles to Simulator
        #self.load_obstacles()




        self.offline_eval = False #self.config.get(
          #  'load_scene_episode_config', False)

    '''
        # scene_episode_config_path = None# self.config.get(
          #  'scene_episode_config_name', None)
        # Sanity check when loading our pre-sampled episodes
        # Make sure the task simulation configuration does not conflict
        # with the configuration used to sample our episode


        # if self.offline_eval:
        #     path = scene_episode_config_path
        #     self.episode_config = \
        #         SocialNavEpisodesConfig.load_scene_episode_config(path)
        #     if self.num_pedestrians != self.episode_config.num_pedestrians:
        #         raise ValueError("The episode samples did not record records for more than {} pedestrians".format(
        #             self.num_pedestrians))
        #     if env.scene.scene_id != self.episode_config.scene_id:
        #         raise ValueError("The scene to run the simulation in is '{}' from the " " \
        #                         scene used to collect the episode samples".format(
        #             env.scene.scene_id))
        #     if self.orca_radius != self.episode_config.orca_radius:
        #         print("value of orca_radius: {}".format(
        #               self.episode_config.orca_radius))
        #         raise ValueError("The orca radius set for the simulation is {}, which is different from "
        #                          "the orca radius used to collect the pedestrians' initial position "
        #                          " for our samples.".format(self.orca_radius))


# class SocialNavRandomTask(PointNavRandomTask):
#     """
#     Social Navigation Random Task
#     The goal is to navigate to a random goal position, in the presence of pedestrians
#     """

#     def __init__(self, env):
#         super(SocialNavRandomTask, self).__init__(env)

#         # Detect pedestrian collision
#         self.termination_conditions.append(PedestrianCollision(self.config))

#         # Decide on how many pedestrians to load based on scene size
#         # Each pixel is 0.01 square meter
#         num_sqrt_meter = env.scene.floor_map[0].nonzero()[0].shape[0] / 100.0
#         self.num_sqrt_meter_per_ped = self.config.get(
#             'num_sqrt_meter_per_ped', 8)
#         self.num_pedestrians = self.config.get('num_pedestrians', 1)
#         # max(1, int(
#         #     num_sqrt_meter / self.num_sqrt_meter_per_ped))

#         """
#         Parameters for our mechanism of preventing pedestrians to back up.
#         Instead, stop them and then re-sample their goals.

#         num_steps_stop         A list of number of consecutive timesteps
#                                each pedestrian had to stop for.
#         num_steps_stop_thresh  The maximum number of consecutive timesteps
#                                the pedestrian should stop for before sampling
#                                a new waypoint.
#         neighbor_stop_radius   Maximum distance to be considered a nearby
#                                a new waypoint.
#         backoff_radian_thresh  If the angle (in radian) between the pedestrian's
#                                orientation and the next direction of the next
#                                goal is greater than the backoffRadianThresh,
#                                then the pedestrian is considered backing off.
#         """
#         self.num_steps_stop = [0] * self.num_pedestrians
#         self.neighbor_stop_radius = self.config.get(
#             'neighbor_stop_radius', 1.0)
#         # By default, stop 2 seconds if stuck
#         self.num_steps_stop_thresh = self.config.get(
#             'num_steps_stop_thresh', 20)
#         # backoff when angle is greater than 135 degrees
#         self.backoff_radian_thresh = self.config.get(
#             'backoff_radian_thresh', np.deg2rad(135.0))

#         """
#         Parameters for ORCA

#         timeStep        The time step of the simulation.
#                         Must be positive.
#         neighborDist    The default maximum distance (center point
#                         to center point) to other agents a new agent
#                         takes into account in the navigation. The
#                         larger this number, the longer the running
#                         time of the simulation. If the number is too
#                         low, the simulation will not be safe. Must be
#                         non-negative.
#         maxNeighbors    The default maximum number of other agents a
#                         new agent takes into account in the
#                         navigation. The larger this number, the
#                         longer the running time of the simulation.
#                         If the number is too low, the simulation
#                         will not be safe.
#         timeHorizon     The default minimal amount of time for which
#                         a new agent's velocities that are computed
#                         by the simulation are safe with respect to
#                         other agents. The larger this number, the
#                         sooner an agent will respond to the presence
#                         of other agents, but the less freedom the
#                         agent has in choosing its velocities.
#                         Must be positive.
#         timeHorizonObst The default minimal amount of time for which
#                         a new agent's velocities that are computed
#                         by the simulation are safe with respect to
#                         obstacles. The larger this number, the
#                         sooner an agent will respond to the presence
#                         of obstacles, but the less freedom the agent
#                         has in choosing its velocities.
#                         Must be positive.
#         radius          The default radius of a new agent.
#                         Must be non-negative.
#         maxSpeed        The default maximum speed of a new agent.
#                         Must be non-negative.
#         """
#         self.neighbor_dist = self.config.get('orca_neighbor_dist', 5)
#         self.max_neighbors = self.num_pedestrians
#         self.time_horizon = self.config.get('orca_time_horizon', 2.0)
#         self.time_horizon_obst = self.config.get('orca_time_horizon_obst', 2.0)
#         self.orca_radius = self.config.get('orca_radius', 0.5)
#         self.orca_max_speed = self.config.get('orca_max_speed', 0.5)

        # self.sim = self.sim

#         # Threshold of pedestrians reaching the next waypoint
#         self.pedestrian_goal_thresh = \
#             self.config.get('pedestrian_goal_thresh', 0.3)
#         self.pedestrians, self.orca_pedestrians = self.load_pedestrians(env)
#         # Visualize pedestrians' next goals for debugging purposes
#         # DO NOT use them during training
#         # self.pedestrian_goals = self.load_pedestrian_goals(env)
#         self.load_obstacles(env)
#         self.personal_space_violation_steps = 0

#         self.offline_eval = self.config.get(
#             'load_scene_episode_config', False)
#         scene_episode_config_path = self.config.get(
#             'scene_episode_config_name', None)
#         # Sanity check when loading our pre-sampled episodes
#         # Make sure the task simulation configuration does not conflict
#         # with the configuration used to sample our episode
#         if self.offline_eval:
#             path = scene_episode_config_path
#             self.episode_config = \
#                 SocialNavEpisodesConfig.load_scene_episode_config(path)
#             if self.num_pedestrians != self.episode_config.num_pedestrians:
#                 raise ValueError("The episode samples did not record records for more than {} pedestrians".format(
#                     self.num_pedestrians))
#             if env.scene.scene_id != self.episode_config.scene_id:
#                 raise ValueError("The scene to run the simulation in is '{}' from the " " \
#                                 scene used to collect the episode samples".format(
#                     env.scene.scene_id))
#             if self.orca_radius != self.episode_config.orca_radius:
#                 print("value of orca_radius: {}".format(
#                       self.episode_config.orca_radius))
#                 raise ValueError("The orca radius set for the simulation is {}, which is different from "
#                                  "the orca radius used to collect the pedestrians' initial position "
#                                  " for our samples.".format(self.orca_radius))
    '''
    def load_obstacles(self) -> None:
        self.map_processor = MapProcessor(self.env.map)
        self.obstacles = self.map_processor.get_obstacles()
        self.sim.processObstacles()

    def load_agents(self) -> None:

        for _ in range(self.num_agents - 1):
            agent = Agent()
            agent.id = self.sim.add_agent(agent.default_pos)
            self.agents[agent.id] = agent
            self.ids += [agent.id]

    '''
        # self.robot_orca_ped = self.sim.addAgent((0, 0))
        # self.orca_pedestrians = []
        # for _ in range(self.num_pedestrians):
        #     orca_ped = self.sim.addAgent((0, 0))
        #     self.orca_pedestrians.append(orca_ped)

    # def load_pedestrians(self) -> None:#, env):
    #     """
    #     Load pedestrians

    #     :param env: environment instance
    #     :return: a list of pedestrians
    #     """
    #     self.robot_orca_ped = self.sim.addAgent((0, 0))
    #     pedestrians = []
    #     orca_pedestrians = []
    #     for _ in range(self.num_pedestrians):
    #         orca_ped = self.sim.addAgent((0, 0))
    #         orca_pedestrians.append(orca_ped)
    #     return pedestrians, orca_pedestrians

    # def load_pedestrian_goals(self, env):
    #     # Visualize pedestrians' next goals for debugging purposes
    #     pedestrian_goals = []
    #     colors = [
    #         [1, 0, 0, 1],
    #         [0, 1, 0, 1],
    #         [0, 0, 1, 1]
    #     ]
    #     for i, ped in enumerate(self.pedestrians):
    #         ped_goal = VisualMarker(
    #             visual_shape=p.GEOM_CYLINDER,
    #             rgba_color=colors[i % 3][:3] + [0.5],
    #             radius=0.3,
    #             length=0.2,
    #             initial_offset=[0, 0, 0.2 / 2])
    #         env.simulator.import_object(ped_goal)
    #         pedestrian_goals.append(ped_goal)
    #     return pedestrian_goals

    # def load_obstacles(self):#, env):
    #     # Add scenes objects to ORCA simulator as obstacles
    #     # for obj_name in env.scene.objects_by_name:
    #     #     obj = env.scene.objects_by_name[obj_name]
    #     #     if obj.category in ['walls', 'floors', 'ceilings']:
    #     #         continue
    #     #     x_extent, y_extent = obj.bounding_box[:2]
    #     #     initial_bbox = np.array([
    #     #         [x_extent / 2.0, y_extent / 2.0],
    #     #         [-x_extent / 2.0, y_extent / 2.0],
    #     #         [-x_extent / 2.0, -y_extent / 2.0],
    #     #         [x_extent / 2.0, -y_extent / 2.0]
    #     #     ])
    #     #     yaw = obj.bbox_orientation_rpy[2]
    #     #     rot_mat = np.array([
    #     #         [np.cos(-yaw), -np.sin(-yaw)],
    #     #         [np.sin(-yaw), np.cos(-yaw)],
    #     #     ])
    #     #     initial_bbox = initial_bbox.dot(rot_mat)
    #     #     initial_bbox = initial_bbox + obj.bbox_pos[:2]
    #     #     self.sim.addObstacle([
    #     #         tuple(initial_bbox[0]),
    #     #         tuple(initial_bbox[1]),
    #     #         tuple(initial_bbox[2]),
    #     #         tuple(initial_bbox[3]),
    #     #     ])

    #     #TODO: Load obstacles from map processor and add it to the simulation

    #     self.sim.processObstacles()
    '''

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

        resample = True
        while resample:

            initial_pos = self.env.get_random_point()
            resample = False

            # # If too close to the robot, resample
            # dist = np.linalg.norm(initial_pos[:2] - self.initial_pos[:2])
            # if dist < self.sim.config.radius:
            #     resample = True
            #     continue

            agent = self.agents[agent_id]

            for neighbor_id in self.ids:
                if neighbor_id == agent_id:
                    continue
                neighbor = self.agents[neighbor_id]
                neighbor_pos = neighbor.get_position()

                dist = np.linalg.norm(np.array(agent.get_position()) -
                                      np.array(neighbor_pos))

                if dist < self.sim.get_radius()*2:
                    resample = True
                    break

        return initial_pos

    def reset_pedestrians(self) -> None:
        """
        Reset the poses of pedestrians to have no collisions with the scene or the robot and set waypoints to follow

        :param env: environment instance
        """
        self.agent_waypoints = {}

        # for ped_id, (ped, orca_ped) in enumerate(zip(self.pedestrians, self.orca_pedestrians)):
        #     pass
        for agent_id, agent in self.agents.items():

            # if self.offline_eval:
            #     episode_index = self.episode_config.episode_index
            #     initial_pos = np.array(
            #         self.episode_config.episodes[episode_index]['pedestrians'][ped_id]['initial_pos'])
            #     initial_orn = np.array(
            #         self.episode_config.episodes[episode_index]['pedestrians'][ped_id]['initial_orn'])
            #     waypoints = self.sample_new_target_pos(
            #         env, initial_pos, ped_id)
            # else:

            #TODO: Get initial position from graph
            initial_pos = self.sample_initial_pos(agent_id)
            # initial_orn = np.array([np.pi / 2.0, 0.0, np.pi / 2.0])
            # initial_orn = p.getQuaternionFromEuler(ped.default_orn_euler)

            waypoints = self.sample_new_target_pos(initial_pos)

            # ped.set_position_orientation(initial_pos, initial_orn)
            self.sim.setAgentPosition(agent, initial_pos)
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
        self.reset_pedestrians()
        self.personal_space_violation_steps = 0

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
            # if self.offline_eval:
            #     if ped_id is None:
            #         raise ValueError(
            #             "The id of the pedestrian to get the goal position was not specified")
            #     episode_index = self.episode_config.episode_index
            #     pos_index = self.episode_config.goal_index[ped_id]
            #     sampled_goals = self.episode_config.episodes[
            #         episode_index]['pedestrians'][ped_id]['target_pos']

            #     if pos_index >= len(sampled_goals):
            #         raise ValueError("The goal positions sampled for pedestrian #{} at "
            #                          "episode {} are exhausted".format(ped_id, episode_index))

            #     target_pos = np.array(sampled_goals[pos_index])
            #     self.episode_config.goal_index[ped_id] += 1
            # else:


            #TODO: Sample or get target position from graph
            target_pos = self.env.get_random_point()

           
            #env.scene.get_random_point(
            # floor=self.floor_num)
            # print('initial_pos', initial_pos)

            #TODO: Get shortest path from graph
            # shortest_path, _ = env.scene.get_shortest_path(
            #     self.floor_num,
            #     initial_pos[:2],
            #     target_pos[:2],
            #     entire_path=True)

            shortest_path = self.env.shortest_path(np.array(initial_pos), np.array(target_pos))
               
            #self.get_shortest_path(initial_pos, target_pos)
            if len(shortest_path) > 1:
                break

        waypoints = self.shortest_path_to_waypoints(shortest_path)
        return waypoints

    def shortest_path_to_waypoints(self, shortest_path):
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
    
    #TODO: FINISH THE REST OF THE FUNCTIONS

    def step(self, env):
        """
        Perform task-specific step: move the pedestrians based on ORCA while
        disallowing backing up

        :param env: environment instance
        """
        # super(SocialNavRandomTask, self).step(env)

        # self.sim.setAgentPosition(
        #     self.robot_orca_ped,
        #     tuple(env.robots[0].get_position()[0:2]))

        #TODO: Remove self.pedestrians from everything and instead get everything from simulator


        # for i, (ped, orca_ped, waypoints) in \
        #         enumerate(zip(self.pedestrians,
        #                       self.orca_pedestrians,
        #                       self.agent_waypoints)):
        #     pass
            
        for agent_id in self.ids:
            agent = self.agents[agent_id]
            agent_pos = agent.get_position()
            waypoints = self.agent_waypoints[agent_id]
            if len(waypoints) == 0:
                raise ValueError("The waypoints for agent {} is empty".format(
                    agent_id))
            
            current_pos = np.array(agent_pos)

            # Sample new waypoints if empty OR
            # if the pedestrian has stopped for self.num_steps_stop_thresh steps
            if len(waypoints) == 0 or \
                    self.num_steps_stop[agent_id] >= self.num_steps_stop_thresh:
                # if self.offline_eval:
                #     waypoints = self.sample_new_target_pos(env, current_pos, i)
                # else:

                waypoints = self.sample_new_target_pos(env, current_pos)
                self.agent_waypoints[agent_id] = waypoints
                self.num_steps_stop[agent_id] = 0

            next_goal = waypoints[0]
            # self.pedestrian_goals[i].set_position(
            #     np.array([next_goal[0], next_goal[1], current_pos[2]]))

            # yaw = np.arctan2(next_goal[1] - current_pos[1],
            #                  next_goal[0] - current_pos[0])
            # agent.set_yaw(yaw)
            
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


        # for i, (agent, orca_pred, waypoints) in \
        #         enumerate(zip(self.pedestrians,
        #                       self.orca_pedestrians,
        #                       self.agent_waypoints)):
            
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

        # next_peds_pos_xyz = \
        #     {i: ped.get_position() for i, ped in enumerate(self.pedestrians)}
        # next_peds_stop_flag = [False for i in range(len(self.pedestrians))]

        # for i, (ped, orca_ped, waypoints) in \
        #         enumerate(zip(self.pedestrians,
        #                       self.orca_pedestrians,
        #                       self.agent_waypoints)):
        #     pass
    
        for agent_id in self.ids:

            agent = self.agents[agent_id]
            pos_xy = self.sim.get_agent_position(agent_id)
            prev_pos_xy = np.array(agent.get_position())
            next_pos_xy = np.array([pos_xy[0], pos_xy[1]])


            # prev_pos_xyz = ped.get_position()
            # next_pos_xyz = np.array([pos_xy[0], pos_xy[1], prev_pos_xyz[2]])

            if self.detect_backoff(agent):
                self.stop_neighbor_pedestrians(agent_id,
                                               next_peds_stop_flag,
                                               next_peds_pos_xy)
            elif next_peds_stop_flag[agent_id] is False:
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


