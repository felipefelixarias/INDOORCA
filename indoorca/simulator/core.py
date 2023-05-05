from typing import List, NamedTuple, Tuple, Dict

import numpy as np

from .orca import IndoorORCASim, IndoorORCASimConfig
from indoorca.environment.core import Environment
import indoorca
from .position import Position

class Agent():


    def __init__(self, map_shape: Tuple[int, int]):

        self.default_pos = [0,0]
        self.pos = self.default_pos
        self.pos_new = Position(map_shape)
        self.id = 0
        self.start_position = self.default_pos
        self.goal_position = self.default_pos

    def set_start_position(self, pos):
        self.start_position = pos

    def set_goal_position(self, pos):
        self.goal_position = pos

    def set_position(self, pos):
        self.pos = pos

    def get_position(self):
        return self.pos

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
        self.obstacles = []
        self.num_agents = num_agents
        self.num_steps_stop = [0] * self.num_agents
        self.neighbor_stop_radius =  1.0
        # By default, stop 2 seconds if stuck
        self.num_steps_stop_thresh = 20
        # backoff when angle is greater than 135 degrees
        self.backoff_radian_thresh = np.deg2rad(135.0)
        self.pedestrian_goal_thresh = indoorca.radius_meters*1.5
        self.personal_space_violation_steps = 0
        self.arrived_at_goal = {}
        self.starts = {}
        self.goals = {}


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

        #TODO: Load obstacles to Simulator
        self.load_obstacles()
        self.load_agents()

    def reset_sim(self, num_agents) -> None:
        """
        Reset the simulator
        """
        self.sim = IndoorORCASim(IndoorORCASimConfig())
        self.sim.trajectories = []
        self.num_agents = num_agents
        self.agents = {}
        self.ids = []
        self.agent_waypoints = {}
        self.obstacles = []
        self.num_agents = num_agents
        self.personal_space_violation_steps = 0
        self.arrived_at_goal = {}
        self.starts = {}
        self.goals = {}

        self.sim.add_obstacles(self.env.get_obstacle_meters())
        self.sim.process_obstacles()
        
        self.load_agents()


    def load_obstacles(self) -> None:
        self.env.process_obstacles()
        obs_list = self.env.get_obstacle_meters()
        self.obstacles = self.sim.add_obstacles(obs_list)
        self.sim.process_obstacles()

    def load_agents(self) -> None:

        for _ in range(self.num_agents):
            agent = Agent(self.env.obstacle_map.shape)
            agent.id = self.sim.add_agent(agent.default_pos)
            self.agents[agent.id] = agent
            self.ids += [agent.id]
            self.arrived_at_goal[agent.id] = False
            self.starts[agent.id] = agent.default_pos
            self.goals[agent.id] = agent.default_pos


    def reset_pedestrians(self) -> None:
        """
        Reset the poses of pedestrians to have no collisions with the scene or the robot and set waypoints to follow

        :param env: environment instance
        """
        self.agent_waypoints = {}

        for agent_id, agent in self.agents.items():

            #TODO: Get initial position from graph
            initial_pos = self.sample_initial_pos(agent_id)
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

        self.personal_space_violation_steps = 0
        self.sim.trajectories = []
        self.sim = IndoorORCASim(IndoorORCASimConfig())
        self.sim.add_obstacles(self.env.get_obstacle_meters())
        self.sim.process_obstacles()
        self.agents = {}
        self.ids = []
        self.load_agents()   
        self.reset_pedestrians()
    
    def get_waypoints(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Get waypoints from start to goal using A* search.

        :param start: start position
        :param goal: goal position
        :return waypoints: list of waypoints from start to goal
        """

        shortest_path, dist = self.env.shortest_path(np.array(start), np.array(goal))
        return shortest_path.tolist()
    
    def get_trajectories(self) -> List[List[List[float]]]:
        """
        Get trajectories of all agents.

        :return trajectories: list of trajectories of all agents
        """
        return self.sim.trajectories
    
    def set_starts_and_goals(self, starts: Dict[int, Tuple[float, float]], 
                             goals: Dict[int, Tuple[float, float]]) -> None:
        """Set the starts and goals for each agent.

        Parameters
        ----------
        starts : 
            Dictionary defining the start positions for each agent.
        goals :
            Dictionary defining the goal positions for each agent.
        """

        #Assert that each agent has start and goal
        assert sorted(starts.keys()) == sorted(goals.keys())

        self.starts = starts
        self.goals = goals

        for agent_id, agent in self.agents.items():
            self.sim.set_agent_position(agent_id, self.starts[agent_id])
            waypoints = self.get_waypoints(self.starts[agent_id], self.goals[agent_id])
            self.agent_waypoints[agent_id] = waypoints


    
    def get_rand_starts_and_goals(self, dist_thresh: float = 4.0, num_samples: int = 100) -> \
                             Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
        
        starts = {}
        goals = {}

        for agent_id, agent in self.agents.items():
            rand_pos = self.env.get_random_point()
            #Make sure the random position is not close to an existing start position
            not_free_pos = True
            while not_free_pos:
                not_free_pos = False
                for start_pos in starts.values():
                    dist = np.linalg.norm(np.array(rand_pos) - np.array(start_pos))
                    if dist < self.pedestrian_goal_thresh:
                        rand_pos = self.env.get_random_point()
                        not_free_pos = True
                        break
            starts[agent_id] = rand_pos
            agent.set_start_position(rand_pos)

        for agent_id, agent in self.agents.items():
            start_pos = starts[agent_id]

            # Find a goal position that is not too close to other occupied positions and
            # at least dist_thresh away from the start position
            valid_goal = False
            while not valid_goal:
                rand_pos = self.env.get_random_point()

                # Check if the random position is not too close to existing occupied positions
                not_free_pos = False
                occupied_positions = list(starts.values()) + list(goals.values())
                for occ_pos in occupied_positions:
                    dist = np.linalg.norm(np.array(rand_pos) - np.array(occ_pos))
                    if dist < self.pedestrian_goal_thresh:
                        not_free_pos = True
                        break

                # If the random position is not too close to occupied positions, check if it's
                # at least dist_thresh away from the start position using shortest_path
                if not not_free_pos:
                    max_dist = 0
                    furthest_sample = None
                    for _ in range(num_samples):
                        sampled_pos = self.env.get_random_point()
                        _, path_dist = self.env.shortest_path(np.array(start_pos), np.array(sampled_pos))

                        if path_dist > dist_thresh:
                            goals[agent_id] = sampled_pos
                            agent.set_goal_position(sampled_pos)
                            valid_goal = True
                            break

                        if path_dist > max_dist:
                            max_dist = path_dist
                            furthest_sample = sampled_pos

                    if not valid_goal:
                        goals[agent_id] = furthest_sample
                        agent.set_goal_position(furthest_sample)
                        valid_goal = True

        self.set_starts_and_goals(starts, goals)
        return starts, goals

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
                t_pos = Position(self.env.obstacle_map.shape)
                t_pos.set_position_pix(target_pos[0], target_pos[1])
                target_pos = (t_pos.x, t_pos.y)
                
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

    
    def step(self) -> bool:

        for agent_id, agent in self.agents.items():
            
            agent_pos = self.sim.get_agent_position(agent_id)
            waypoints = self.agent_waypoints[agent_id]
            current_pos = agent_pos

            #Check if agent has reached goal
            if np.linalg.norm(np.array(agent_pos) - np.array(self.goals[agent_id])) < self.pedestrian_goal_thresh:
                print('Agent {} reached goal'.format(agent_id))
                temp = self.goals[agent_id]
                self.goals[agent_id] = self.starts[agent_id]
                self.starts[agent_id] = temp

                self.agent_waypoints[agent_id] = self.get_waypoints(current_pos, self.goals[agent_id])
                self.arrived_at_goal[agent_id] = True

            #Resample for waypoints
            if len(waypoints) == 0:
                goal_pos = self.goals[agent_id]
                self.agent_waypoints[agent_id] = self.get_waypoints(current_pos, goal_pos)   

                waypoints = self.agent_waypoints[agent_id]

            #TODO: Implement backing up detection

            next_goal = self.agent_waypoints[agent_id][0]
            desired_velocity = np.array(next_goal) - np.array(current_pos)
            desired_velocity = (desired_velocity / np.linalg.norm(desired_velocity)) * self.sim.get_max_speed()
            self.sim.set_agent_pref_velocity(agent_id, list(desired_velocity))

        if np.all([self.arrived_at_goal[agent_id] for agent_id in self.agents.keys()]):
            print('All agents have reached their goals')
            return True

        self.sim.do_step()
    
        #Check if agent has reached waypoint, if so pop it
        for agent_id, agent in self.agents.items():
            agent_pos = self.sim.get_agent_position(agent_id)
            next_goal = self.agent_waypoints[agent_id][0]
            if np.linalg.norm(np.array(agent_pos) - np.array(next_goal)) < self.pedestrian_goal_thresh:
                self.agent_waypoints[agent_id].pop(0)

        return False
    
    def run(self, num_rounds: int = 1, max_steps: int = 500) -> None:
        """Function to run the simulation

        Parameters
        ----------
        num_rounds : optional
            The number of times the agents should try to reach their goals, by default 1
        max_steps : optional
            The maximum number of timesteps to simulate, by default 500
        """        

        done = False
        for i in range(max_steps):
            done = self.step()
            if done:
                print('Done at step {}'.format(i))
                num_rounds -= 1
                self.arrived_at_goal = {agent_id: False for agent_id in self.agents.keys()}

            if num_rounds == 0:
                break

            if i%100 == 0:
                print('Step {}'.format(i))     
            
            if i == max_steps - 1:
                print('Max steps reached')

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
    