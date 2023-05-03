import logging
from typing import List, NamedTuple, Tuple

import numpy as np
import rvo2

from indoorca.environment.core import Environment

class IndoorORCASimConfig(NamedTuple):
    """ A wrapper for RVO2 simulator with additional functions for processing a binary traversability map and navigation. 
    
    Parameters
    ----------
    time_step :
        The time step of the simulation. Must be positive
    neighbor_dist :
        The default maximum distance (center point to center point) to other agents a agent takes into account 
        in the navigation. The larger this number, the longer the running time of the simulation. If the number 
        is too low, the simulation will not be safe. Must be non-negative
    max_neighbors :
        The default maximum number of other agents a agent takes into account in the navigation. The larger 
        this number, the longer the running time of the simulation. If the number is too low, the simulation 
        will not be safe. Must be non-negative
    time_horizon :
        The default minimum amount of time for which a agent's velocities that are computed by the simulation 
        are safe with respect to other agents. The larger this number, the sooner a agent will respond to the 
        presence of other agents, but the less freedom a agent has in choosing its velocities. Must be positive
    time_horizon_obst :
        The default minimum amount of time for which a agent's velocities that are computed by the simulation 
        are safe with respect to obstacles. The larger this number, the sooner a agent will respond to the 
        presence of obstacles, but the less freedom a agent has in choosing its velocities. Must be positive
    radius :
        The default radius of a agent. Must be non-negative
    max_speed :
        The default maximum speed of a agent. Must be non-negative
    velocity :
        The default initial two-dimensional linear velocity of a agent (optional)
    """
    
    time_step: float = 1/32.
    neighbor_dist: float = 5.
    max_neighbors: int = 4
    time_horizon: float = 1.25
    time_horizon_obst: float = 2.
    radius: float = 0.125
    max_speed: float = 0.5
    # velocity: Tuple[float] = (0.,0.)


class IndoorORCASim:
    """ Wrapper for RVO2 simulator with additional functions for processing a binary traversability map and navigation.
    
    Parameters
    ----------
    config :
        Configuration for the simulator

    Attributes
    ----------
    sim :
        RVO2 simulator
    """
    def __init__(self, config: IndoorORCASimConfig):
        self.config = config
        self.sim = rvo2.PyRVOSimulator(config.time_step, config.neighbor_dist, config.max_neighbors, config.time_horizon,
                                       config.time_horizon_obst, config.radius, config.max_speed)

        self.environment = None
        self.obstacles = None
        self._no_steps = True
        self.trajectories = []

    def get_time_step(self) -> float:
        """ Returns the time step of the simulation.
        
        Returns
        -------
        float :
            The time step of the simulation
        """
        return self.config.time_step
    
    def get_neighbor_dist(self) -> float:
        """ Returns the default maximum distance (center point to center point) to other agents a agent takes into account 
        in the navigation.
        
        Returns
        -------
        float :
            The default maximum distance to other agents a agent takes into account in the navigation
        """
        return self.config.neighbor_dist
    
    def get_max_neighbors(self) -> int:
        """ Returns the default maximum number of other agents a agent takes into account in the navigation.
        
        Returns
        -------
        int :
            The default maximum number of other agents a agent takes into account in the navigation
        """
        return self.config.max_neighbors
    
    def get_time_horizon(self) -> float:
        """ Returns the default minimum amount of time for which a agent's velocities that are computed by the simulation 
        are safe with respect to other agents.
        
        Returns
        -------
        float :
            The default minimum amount of time for which a agent's velocities that are computed by the simulation 
            are safe with respect to other agents
        """
        return self.config.time_horizon
    
    def get_time_horizon_obst(self) -> float:
        """ Returns the default minimum amount of time for which a agent's velocities that are computed by the simulation 
        are safe with respect to obstacles.
        
        Returns
        -------
        float :
            The default minimum amount of time for which a agent's velocities that are computed by the simulation 
            are safe with respect to obstacles
        """
        return self.config.time_horizon_obst
    
    def get_radius(self) -> float:
        """ Returns the default radius of a agent.
        
        Returns
        -------
        float :
            The default radius of a agent
        """
        return self.config.radius
    
    def get_max_speed(self) -> float:
        """ Returns the default maximum speed of a agent.
        
        Returns
        -------
        float :
            The default maximum speed of a agent
        """
        return self.config.max_speed
    
    # def get_velocity(self) -> Tuple[float]:
    #     """ Returns the default initial two-dimensional linear velocity of a agent.
        
    #     Returns
    #     -------
    #     Tuple[float] :
    #         The default initial two-dimensional linear velocity of a agent
    #     """
    #     return self.config.velocity
    
    def get_obstacles(self) -> List[List[float]]:
        """ Returns the obstacles of the simulation.
        
        Returns
        -------
        List[List[float]] :
            The obstacles of the simulation
        """
        return self.obstacles

    def reset(self, environment: Environment) -> None:
        """ Resets the simulation.

        Parameters
        ----------
        environment :
            The environment to be simulated
        """
        self.environment = environment
        # self.sim.clearAgents()
        # self.sim.clearObstacles()
        self._no_steps = True
        self.trajectories = []

    def soft_reset(self) -> None:
        """ Resets the simulation without clearing the obstacles.

        """
        for i in range(self.get_num_agents()):
            self.sim.setAgentPosition(i, self.get_agent_position(i))

        self._no_steps = True
        self.trajectories = []



    def add_agent(self, position: List[float], velocity: List[float] = []) -> int:
        """ Adds an agent to the simulation.
        
        Parameters
        ----------
        position :
            The two-dimensional starting position of this agent
        velocity : 
            The two-dimensional starting velocity of this agent (optional)

        Returns
        -------
        int : 
            The number of the agent
        """
        return self.sim.addAgent(tuple(position))
    
    def add_obstacle(self, vertices: List[List[float]]) -> int:
        """ Adds an obstacle to the simulation.
        
        Parameters
        ----------
        vertices : List[List[float]]
            A list of vertices of the polygonal obstacle in counterclockwise order

        Returns
        -------
        int :
            The number of the obstacle
        """
        return self.sim.addObstacle(vertices)
    
    def add_obstacles(self, obstacles: List[List[List[float]]]) -> List[int]:
        """ Adds a list of obstacles to the simulation.
        
        Parameters
        ----------
        obstacles : List[List[List[float]]]
            A list of obstacles, where each obstacle is a list of vertices of the polygonal obstacle in counterclockwise order

        Returns
        -------
        List[int] :
            The numbers of the obstacles
        """
        return [self.add_obstacle(obstacle) for obstacle in obstacles]
    
    def process_obstacles(self) -> None:
        """ Processes the obstacles that have been added so that they are accounted for in the simulation.
        """
        self.sim.processObstacles()

    def set_agent_pref_velocity(self, agent_no: int, pref_velocity: List[float]) -> None:
        """ Sets the preferred velocity of a agent.
        
        Parameters
        ----------
        agent_no :
            The number of the agent whose preferred velocity is to be set
        pref_velocity :
            The two-dimensional preferred velocity of the agent
        """
        self.sim.setAgentPrefVelocity(agent_no, tuple(pref_velocity))

    def set_agent_position(self, agent_no: int, position: List[float]) -> None:
        """ Sets the position of a agent.
        
        Parameters
        ----------
        agent_no :
            The number of the agent whose position is to be set
        position :
            The two-dimensional position of the agent
        """
        self.sim.setAgentPosition(agent_no, tuple(position))

    def get_num_agents(self) -> int:
        """ Returns the number of agents in the simulation.
        
        Returns
        -------
        int :
            The number of agents in the simulation
        """
        return self.sim.getNumAgents()
    
    def get_num_obstacles(self) -> int:
        """ Returns the number of obstacles in the simulation.
        
        Returns
        -------
        int :
            The number of obstacles in the simulation
        """
        return self.sim.getNumObstacles()
    
    def get_num_obstacle_vertices(self) -> int:
        """ Returns the number of obstacle vertices.

        Returns
        -------
        int :
            The number of vertices of the obstacle
        """
        return self.sim.getNumObstacleVertices()
    
    def get_agent_position(self, agent_no: int) -> List[float]:
        """ Returns the position of a agent.
        
        Parameters
        ----------
        agent_no :
            The number of the agent whose position is to be retrieved

        Returns
        -------
        List[float] :
            The two-dimensional position of the agent
        """
        return self.sim.getAgentPosition(agent_no)
    
    def get_agent_velocity(self, agent_no: int) -> List[float]:
        """ Returns the velocity of a agent.
        
        Parameters
        ----------
        agent_no :
            The number of the agent whose velocity is to be retrieved

        Returns
        -------
        List[float] :
            The two-dimensional velocity of the agent
        """
        return self.sim.getAgentVelocity(agent_no)
    
    def get_global_time(self) -> float:
        """ Returns the global time of the simulation.
        
        Returns
        -------
        float :
            The global time of the simulation
        """
        return self.sim.getGlobalTime()
    
    def do_step(self) -> None:
        """ Performs a simulation step and updates the two-dimensional position and two-dimensional velocity of each agent.
        """
        #Add current positions to trajectories, if first step, loop over all agents and create new trajectory
        if self._no_steps:
            for i in range(self.get_num_agents()):
                self.trajectories.append([])
                self.trajectories[i].append(list(self.get_agent_position(i)))
            self._no_steps = False
        else:
            for i in range(self.get_num_agents()):
                self.trajectories[i].append(list(self.get_agent_position(i)))

        self.sim.doStep()
    

    




    

