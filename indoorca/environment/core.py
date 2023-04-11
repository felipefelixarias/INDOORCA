import logging
from typing import List, Tuple

import numpy as np
import networkx as nx
import cv2
from PIL import Image

import indoorca
from indoorca.utils.math import l2_distance

class Environment:

    """Class containing all information relevant to a single environment

    Parameters
    ----------
    name
        Name of the environment
    map
        Binary map of the environment

    Attributes
    ----------
    name
        See above
    map
        See above
    trav_map
        Processed map that accounts for the agent radius and empty space
    map_size
        Size of the map in pixels
    graph
        Graph representing the floor plan
    """
    def __init__(self, 
        name: str,
        map: np.ndarray
        )-> None:

        self.name = name
        self.map = self._trim_map(np.asarray(map))
        self.map_size = self.map.shape
        self.trav_map = self._get_trav_map()
        self.graph = None

    def build_graph(self) -> Tuple[nx.Graph, np.ndarray]:
        """Build a nx.Graph from the traversability map

        Returns:
            nx.Graph
                Graph representing the floor plan
            
        """    
        g = nx.Graph()
        for i in range(self.trav_map.shape[0]):
            for j in range(self.trav_map.shape[1]):
                if self.trav_map[i, j] == 0:
                    continue
                g.add_node((i, j))
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for n in neighbors:
                    if 0 <= n[0] < self.trav_map.shape[0] and \
                    0 <= n[1] < self.trav_map.shape[1] and \
                    self.trav_map[n[0], n[1]] > 0:
                        g.add_edge(n, (i, j), weight=l2_distance(n, (i, j)))

        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
        reachable_trav_map = np.zeros(self.trav_map.shape)
        for node in g.nodes:
            reachable_trav_map[node[0], node[1]] = indoorca.free_space

        self.trav_map = reachable_trav_map
        return g

    def _map_to_world(self, xy: np.ndarray) -> np.ndarray:
        """Convert map coordinates to world coordinates

        Args:
            xy
                Map coordinates

        Returns:
            np.ndarray
                World coordinates
        """        

        axis = 0 if len(xy.shape) == 1 else 1
        return np.flip((xy - np.array(self.trav_map.shape)/2.0)
                        /indoorca.pix_per_meter, axis=axis)

    def _world_to_map(self, xy: np.ndarray) -> np.ndarray:
        """Convert world coordinates to map coordinates

        Args:
            xy
                World coordinates

        Returns:
            np.ndarray
                Map coordinates
        """        
        return np.flip(xy*indoorca.pix_per_meter + 
                        np.array(self.trav_map.shape)/2.0).astype(int)

    def shortest_path(self, 
                      source_world: np.ndarray, 
                      target_world: np.ndarray,
                      entire_path: bool = False
                      ) -> Tuple[List[np.ndarray], float]:
        """Find the shortest path between two points

        Args:
            source_world
                Source point in world coordinates
            target_world
                Target point in world coordinates
            entire_path
                If True, return the entire path. If False, return only waypoints
        
        Returns:
            List[np.ndarray]
                List of waypoints in world coordinates
            float
                Length of the path  
        """

        source = tuple(self._world_to_map(source_world))
        target = tuple(self._world_to_map(target_world))

        if not self.graph.has_node(source):
            nodes = np.array(self.graph.nodes)
            closest_node = tuple(
                nodes[np.argmin(np.linalg.norm(nodes - source, axis=1))])
            self.graph.add_edge(closest_node, source,
                        weight=l2_distance(source, closest_node))

        if not self.graph.has_node(target):
            nodes = np.array(self.graph.nodes)
            closest_node = tuple(
                nodes[np.argmin(np.linalg.norm(nodes - target, axis=1))])
            self.graph.add_edge(closest_node, target, 
                        weight=l2_distance(target, closest_node))

        path_map = np.asarray(nx.astar_path(
            self.graph, source, target, heuristic=l2_distance))

        path_world = self._map_to_world(path_map)
        geodesic_dist = np.sum(np.linalg.norm(path_world[1:]-path_world[:-1], axis=1))

        return path_world, geodesic_dist        

    def _trim_map(self, map: np.ndarray) -> np.ndarray:
        """Trim empty space around the map

        Args:
            map
                Map to trim

        Returns:
            np.ndarray
                Trimmed map
        """        
        #Change all obstacles to obstacle_space global variable
        map[map>0] = indoorca.free_space
        map[map==0] = indoorca.obstacle_space

        #TODO: Trim maps once data is simulated by OPAM

        #Check if map is all free space
        if not np.all(map==indoorca.free_space):
            coords = np.argwhere(map>0)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            trimmed_map = map[x_min:x_max+1, y_min:y_max+1]
        else:
            trimmed_map = map

        # Add one meter of obstacle around the map
        # trimmed_map = np.pad(trimmed_map, indoorca.pix_per_meter,
        #                     'constant', constant_values=indoorca.obstacle_space) 

        return trimmed_map


    def _get_trav_map(self) -> np.ndarray:
        """Process map to account for agent radius

        Returns:
            np.ndarray
                Traversability map with obstacles and free space
        """   
        kernel = np.ones((indoorca.max_agent_radius, indoorca.max_agent_radius)).astype(np.uint8)
        trav_map = cv2.erode(self.map.astype('uint8'), kernel)

        return trav_map
    
