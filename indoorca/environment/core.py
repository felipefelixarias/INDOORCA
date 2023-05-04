import logging
from typing import List, Tuple

import numpy as np
import networkx as nx
import cv2
from matplotlib.patches import Polygon as mplPolygon
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.path import Path
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon, Point
import bisect
import random

from indoorca.simulator.position import Position



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
        self.trav_map = None
        self.graph = None
        self.obstacles_meters = []
        self.obstacles_pixels = []
        self._obstacles_polygons = []
        self.obstacle_map = None
        self.pix_per_meter = indoorca.pix_per_meter
        self.waypoint_dist = 0.1
        self.num_waypoints = 4
        

    def get_obstacle_meters(self)->List[List[List[float]]]:
        """Return the obstacle coordinates in meters

        Returns
        -------
        List[List[List[float]]]
            Obstacle coordinates in meters
        """
        return self.obstacles_meters
    
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



    def process_obstacles(self)->None:
        """Process the obstacles in the environment and store them in the
        obstacles attribute
        """
        _, reachable_trav_map = self.build_graph(self.map)
        self._obstacles_polygons = self._get_obstacle_polygons(reachable_trav_map)
        self._get_obstacle_meters()
        self._construct_obstacle_map()
        self._compute_trav_map()

    def _compute_trav_map(self)->None:
        """Compute the traversability map for the environment
        """
        radius = int(indoorca.pix_per_meter * 0.55)
        kernel = np.ones((radius, radius)).astype(np.uint8)
        eroded_map = cv2.erode(self.obstacle_map.astype('uint8'), kernel)
        self.graph, self.trav_map = self.build_graph(eroded_map)

    def build_graph(self, map: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
        """Compute the largest connected component of the map provided and return
        the graph and the processed map

        Parameters
        ----------
        map : 
            Binary map of the environment

        Returns
        -------
        Tuple[nx.Graph, np.ndarray]
            Graph representing the floor plan and processed map
        """        

        g = nx.Graph()
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if map[i, j] == indoorca.obstacle_space:
                    continue
                g.add_node((i, j))
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for n in neighbors:
                    if 0 <= n[0] < map.shape[0] and \
                    0 <= n[1] < map.shape[1] and \
                    map[n[0], n[1]] > 0:
                        g.add_edge(n, (i, j), weight=l2_distance(n, (i, j)))

        # Get the largest connected component
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
        reachable_trav_map = np.zeros(map.shape)
        for node in g.nodes:
            reachable_trav_map[node[0], node[1]] = indoorca.free_space
        
        return g, reachable_trav_map
    
    def get_random_point(self) -> List[float]:
        """Return a random point in the environment

        Returns
        -------
        List[int]
            Random point in the environment
        """     

        #Check if the environment has been processed
        if self.trav_map is None:
            raise ValueError("Environment has not been processed yet")

        #Check if the environment is all obstacle space
        if np.all(self.trav_map == indoorca.obstacle_space):
            raise ValueError("Environment is all obstacle space")
        
        # print('environment shape:', self.trav_map.shape)

        #Get random point from the traversability graph
        all_locations = list(self.graph.nodes)

        point = list(all_locations[np.random.randint(0, len(all_locations))])
        print('random point:', point)
        #Convert to meters
        point = self._map_to_world(np.array(point))
        # print('random point in meters:', point)

        return [point[0], point[1]]
          
        # while True:



        #     point = [np.random.randint(0, self.trav_map.shape[0]), np.random.randint(0, self.trav_map.shape[1])]
        
        #     if self.trav_map[point[0], point[1]] == indoorca.free_space:
        #         #Convert to meters
        #         point = self._map_to_world(np.array(point))
        #         return point.tolist()
            
        

    def _simplify_polygon(self, polygon: Polygon, tolerance: float=0.02) -> List[Polygon]:
        """ Return the simplified polygon.
        
        Parameters
        ----------
        polygon :
            Polygon to be simplified
        tolerance :
            Tolerance. Default is 0.0.
        
        Returns
        -------
        Polygon :
            Simplified polygon
        """
        simplified = polygon.simplify(0.02, preserve_topology=True)
        if isinstance(simplified, Polygon):
            return [simplified]
        return list(simplified)
    
    def _get_obstacle_polygons(self, 
                               binary_map: np.ndarray,
                               tolerance: float=0.02, 
                               min_area_ratio: float=0.0001 
                               )-> List[Polygon]:
        """Process the map and return a list of Polygon objects.

        Parameters
        ----------
        tolerance : optional
            Tolerance for simplifying the polygons, by default 0.02
        min_area_ratio : optional
            The threshold for ignoring obstacles, by default 0.0001

        Returns
        -------
        List[Polygon]
            List of Polygon objects extracted from the map
        """        

        contours = measure.find_contours(binary_map, 0.5)

        polygons = []
        for contour in contours:
            # Convert coordinates back to the original scale
            scaled_contour = [((point[1] * 2) / (binary_map.shape[1] - 1)) - 1 for point in contour]
            scaled_contour_y = [((point[0] * 2) / (binary_map.shape[0] - 1)) - 1 for point in contour]

            # Ensure the contour has at least 3 points to form a valid polygon
            if len(contour) < 3:
                continue

            # Create a Shapely polygon from the contour
            polygon_coords = list(zip(scaled_contour, scaled_contour_y))
            polygon = Polygon(polygon_coords)

            # Discard very small polygons based on the min_area_ratio
            if polygon.area < min_area_ratio:
                continue

            # Simplify the polygon and add it to the list
            simplified_polygons = self._simplify_polygon(polygon, tolerance)
            polygons.extend(simplified_polygons)


        #Get the obstacle vertices in pixel space, they are currently in range (-1,1)
        pixel_coords = []
        for polygon in polygons:
            obstacle = []
            for coord in polygon.exterior.coords:
                #Transform coordinates from range (-1,1) to (0,1)
                coord = np.asarray(coord)
                coord = (coord + 1) / 2
                #Transform coordinates from range (0,1) to (0, width/height)
                coord = coord * np.asarray([binary_map.shape[1]-1, binary_map.shape[0]-1])
                obstacle.append(coord)
            pixel_coords.append(obstacle)

        #Add the polygons to the obstacles list
        self.obstacles_pixels.extend(pixel_coords)
        return polygons


    def _construct_obstacle_map(self)->None:
        """Construct the obstacle map from the polygons
        """
        # Initialize an empty binary map
        height = self.map.shape[0]
        width = self.map.shape[1]

        #Initialize map as all freespace
        obstacle_map = np.ones((height, width)) * indoorca.free_space



        # Iterate through each polygon
        for obs_pixel in self.obstacles_pixels:
            # Convert the meter coordinates to pixels
            np_obs_pixel = np.array([obs_pixel], dtype=np.int32)

            # Fill the obstacle polygon with 0 (non-traversable)
            cv2.fillPoly(obstacle_map, np_obs_pixel, 0)#indoorca.obstacle_space)
            
        self.obstacle_map = obstacle_map        
    
    def _get_obstacle_meters(self) -> None:
        """ Return a list of obstacles for the simulator.
        
        Returns
        -------
        List[List[Tuple[float]]] :
            List of obstacles' vertices for the simulator
        """
        
        for obs_pixel in self.obstacles_pixels:
            obs_meter = self.map_to_world(np.asarray(obs_pixel))
            self.obstacles_meters.append(obs_meter.tolist())


    def display_map_with_polygons(self, polygon_color: str='red')-> None:
        """ Display the binary map with the polygons.

        Parameters
        ----------
        polygon_color : optional
            Color of the polygons, by default 'red'
        """

        _, ax = plt.subplots()

        # Display the binary map
        ax.imshow(self.map, cmap='gray', extent=(-1, 1, -1, 1), origin='lower')

        # Plot the polygons
        for polygon in self._obstacles_polygons:
            coords = list(polygon.exterior.coords)
            coords_tuples = [(x, y) for x, y in coords]

            # Ensure the coordinates have at least 3 points to form a valid polygon
            if len(coords_tuples) < 3:
                continue

            mpl_polygon = mplPolygon(coords_tuples, edgecolor=polygon_color, facecolor='none', linewidth=2)
            ax.add_patch(mpl_polygon)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal', 'box')
        plt.show()  


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
        xy = np.asarray(xy)
        ret = (xy - np.array(self.map.shape)/2.0)/indoorca.pix_per_meter
        ret = np.asarray(ret)
        # print('ret map to world', ret)
        #Check if the points are within the map and it is a single point
        # print('ret.shape', ret.shape)
        # ret = ret.reshape(1, -1)
 
        #TODO: Verify if this is correct and that the conversions aren't messed up

        # if np.any(ret[:, 1] < -self.map.shape[0]/(2.0*indoorca.pix_per_meter)):
        #     raise ValueError("Point is outside the map with a negative value in x.")
        
        # if np.any(ret[:, 1] > self.map.shape[0]/(2.0*indoorca.pix_per_meter)):
        #     raise ValueError("Point is outside the map with a positive value in x.")
        
        # if np.any(ret[:, 0] < -self.map.shape[1]/(2.0*indoorca.pix_per_meter)):
        #     raise ValueError("Point is outside the map with a negative value in y.")
        
        # if np.any(ret[:, 0] > self.map.shape[1]/(2.0*indoorca.pix_per_meter)):
        #     raise ValueError("Point is outside the map with a positive value in y.")
        
        return ret

    def _world_to_map(self, xy: np.ndarray) -> np.ndarray:
        """Convert world coordinates to map coordinates

        Args:
            xy
                World coordinates

        Returns:
            np.ndarray
                Map coordinates
        """        
        xy = np.asarray(xy)
        ret = xy*indoorca.pix_per_meter + np.array(self.map.shape)/2.0
        ret = np.asarray(ret).astype(np.int)
        # print('ret world to map', ret)
        # if np.any(ret < 0):
        #     raise ValueError("Point is outside the map with a value less than zero.")
        
        # if np.any(ret[:, 0] >= self.map.shape[0]):
        #     raise ValueError("Point is outside the map with a value greater than the map size.")
        
        # if np.any(ret[:, 1] >= self.map.shape[1]):
        #     raise ValueError("Point is outside the map with a value greater than the map size.")
        
        return ret
    
    def map_to_world(self, xy):
        """
        Transforms a 2D point in map reference frame into world (simulator) reference frame

        :param xy: 2D location in map reference frame (image)
        :return: 2D location in world reference frame (metric)
        """
        axis = 0 if len(xy.shape) == 1 else 1
        return np.flip((xy - np.array(self.map.shape) / 2.0) * indoorca.pix_per_meter, axis=axis)

    def world_to_map(self, xy):
        """
        Transforms a 2D point in world (simulator) reference frame into map reference frame

        :param xy: 2D location in world reference frame (metric)
        :return: 2D location in map reference frame (image)
        """
        return np.flip((xy / self.trav_map_resolution + self.trav_map_size / 2.0)).astype(np.int)
    
    def approximate_average_shortest_path_length(self, num_samples: int = 1000) -> float:
        nodes = list(self.graph.nodes)
        sampled_nodes = random.sample(nodes, min(num_samples, len(nodes)))

        path_lengths = []
        for source in sampled_nodes:
            for target in sampled_nodes:
                if source != target:
                    length = nx.single_source_dijkstra_path_length(self.graph, source, target, weight='weight')
                    path_lengths.append(length)

        avg_shortest_path_length = sum(path_lengths) / len(path_lengths)
        return avg_shortest_path_length

    def find_goal_at_avg_shortest_path_length(self, source_world: np.ndarray, num_samples: int = 1000, max_attempts: int = 100) -> np.ndarray:
        print("Finding goal at average shortest path length for source: ", source_world)
        source = self._world_to_map(np.array(source_world))
        avg_shortest_path_length = 4.#nx.average_shortest_path_length(self.graph, weight='weight')
        #avg_shortest_path_length = self.approximate_average_shortest_path_length()
        print('Average shortest path length: ', avg_shortest_path_length)
        nodes = list(self.graph.nodes)
        print('source: ', source)
        print('nodes: ', nodes)
        
        distances = [nx.single_source_dijkstra_path_length(self.graph, tuple(source), tuple(target), weight='weight') for target in nodes]

        sorted_distances = sorted(distances)

        print('Sorted distances: ', sorted_distances)
        idx = bisect.bisect_left(sorted_distances, avg_shortest_path_length)

        if idx == len(sorted_distances):
            idx -= 1

        if sorted_distances[idx] == avg_shortest_path_length:
            goal = nodes[distances.index(sorted_distances[idx])]
            goal_world = self._map_to_world(goal)
            return goal_world

        for _ in range(max_attempts):
            sampled_nodes = random.sample(nodes, min(num_samples, len(nodes)))
            sampled_distances = [nx.single_source_dijkstra_path_length(self.graph, source, target, weight='weight') for target in sampled_nodes]

            closest_distance = min(sampled_distances, key=lambda x: abs(x - avg_shortest_path_length))
            if abs(closest_distance - avg_shortest_path_length) < 1e-6:
                goal = sampled_nodes[sampled_distances.index(closest_distance)]
                goal_world = self._map_to_world(goal)
                return goal_world

        raise ValueError("Unable to find a goal at the average shortest path length.")

    # def find_goal_at_avg_shortest_path_length(self, source_world: np.ndarray, num_samples: int = 100) -> np.ndarray:
    #     """Find a goal position that is approximately the average shortest path length away from the source

    #     Args:
    #         source_world
    #             Source point in world coordinates
    #         num_samples
    #             Number of nodes to sample for calculating the shortest path lengths

    #     Returns:
    #         np.ndarray
    #             Goal position in world coordinates
    #     """

    #     source = self._world_to_map(np.array(source_world))
    #     avg_shortest_path_length = nx.average_shortest_path_length(self.graph, weight='weight')

    #     # Sample nodes from the graph
    #     nodes = list(self.graph.nodes)
    #     sampled_nodes = random.sample(nodes, min(num_samples, len(nodes)))

    #     # Calculate the shortest path length from the source to the sampled nodes
    #     path_lengths = {node: nx.dijkstra_path_length(self.graph, source, node, weight='weight') for node in sampled_nodes}

    #     # Sort the path lengths and nodes
    #     sorted_path_lengths = sorted((length, node) for node, length in path_lengths.items())

    #     # Find the index of the path length closest to the average using binary search
    #     closest_goal_index = bisect.bisect_left(sorted_path_lengths, (avg_shortest_path_length,))
    #     closest_goal_index = min(closest_goal_index, len(sorted_path_lengths) - 1)  # Handle edge case

    #     # Check if the index to the left is closer to the average shortest path length
    #     if closest_goal_index > 0:
    #         diff_current = abs(sorted_path_lengths[closest_goal_index][0] - avg_shortest_path_length)
    #         diff_prev = abs(sorted_path_lengths[closest_goal_index - 1][0] - avg_shortest_path_length)
    #         if diff_prev < diff_current:
    #             closest_goal_index -= 1

    #     closest_goal = sorted_path_lengths[closest_goal_index][1]
    #     goal_world = self._map_to_world(closest_goal)

    #     return goal_world


# def find_goal_at_avg_shortest_path_length(self, source_world: np.ndarray) -> np.ndarray:
#     """Find a goal position that is approximately the average shortest path length away from the source

#     Args:
#         source_world
#             Source point in world coordinates

#     Returns:
#         np.ndarray
#             Goal position in world coordinates
#     """

#     source = self._world_to_map(np.array(source_world))
#     avg_shortest_path_length = nx.average_shortest_path_length(self.graph, weight='weight')

#     # Calculate the shortest path length from the source to all nodes
#     path_lengths = nx.single_source_dijkstra_path_length(self.graph, source, weight='weight')

#     # Create a SortedList of the path lengths and nodes
#     sorted_path_lengths = SortedList((length, node) for node, length in path_lengths.items())

#     # Find the index of the path length closest to the average using bisect_left
#     closest_goal_index = sorted_path_lengths.bisect_left((avg_shortest_path_length,))
#     closest_goal_index = min(closest_goal_index, len(sorted_path_lengths) - 1)  # Handle edge case

#     # Check if the index to the left is closer to the average shortest path length
#     if closest_goal_index > 0:
#         diff_current = abs(sorted_path_lengths[closest_goal_index][0] - avg_shortest_path_length)
#         diff_prev = abs(sorted_path_lengths[closest_goal_index - 1][0] - avg_shortest_path_length)
#         if diff_prev < diff_current:
#             closest_goal_index -= 1

#     # Fallback mechanism: if no suitable goal is found, return the farthest node from the start position
#     if sorted_path_lengths[closest_goal_index][0] < avg_shortest_path_length:
#         closest_goal_index = len(sorted_path_lengths) - 1

#     closest_goal = sorted_path_lengths[closest_goal_index][1]
#     goal_world = self._map_to_world(closest_goal)

#     return goal_world
    
    # def find_goal_at_avg_shortest_path_length(self, source_world: np.ndarray) -> np.ndarray:
    #     """Find a goal position that is approximately the average shortest path length away from the source

    #     Args:
    #         source_world
    #             Source point in world coordinates

    #     Returns:
    #         np.ndarray
    #             Goal position in world coordinates
    #     """

    #     print('source_world: ', source_world)
    #     source = self._world_to_map(np.array(source_world))
    #     print('source: ', source)
    #     avg_shortest_path_length = nx.average_shortest_path_length(self.graph, weight='weight')

    #     # Calculate the shortest path length from the source to all nodes
    #     path_lengths = nx.single_source_dijkstra_path_length(self.graph, source, weight='weight')

    #     # Find the node closest to the average shortest path length
    #     closest_goal = None
    #     min_diff = float('inf')
    #     for node, length in path_lengths.items():
    #         diff = abs(length - avg_shortest_path_length)
    #         if diff < min_diff:
    #             min_diff = diff
    #             closest_goal = node

    #     print('closest_goal: ', closest_goal)
    #     goal_world = self._map_to_world(closest_goal)


    #     return goal_world
        
    # def find_goal_at_avg_shortest_path_length(self, source_world: np.ndarray, num_attempts: int = 1000) -> np.ndarray:
    #     """Find a goal position that is approximately the average shortest path length away from the source

    #     Args:
    #         source_world
    #             Source point in world coordinates
    #         num_attempts
    #             Number of attempts to find a goal at the approximate distance

    #     Returns:
    #         np.ndarray
    #             Goal position in world coordinates
    #     """

    #     avg_shortest_path_length = nx.average_shortest_path_length(self.graph, weight='weight')

    #     # Try to find a goal position within a range around the average shortest path length
    #     for _ in range(num_attempts):
    #         random_goal = self.random_position()
    #         random_goal_world = self._map_to_world(random_goal)
    #         path_map = np.asarray(nx.astar_path(self.graph, source_world, random_goal_world, heuristic=l2_distance))
    #         path_length = np.sum(np.linalg.norm(path_map[1:] - path_map[:-1], axis=1))

    #         if abs(path_length - avg_shortest_path_length) < 0.1 * avg_shortest_path_length:  # Tolerance of 10%
    #             return random_goal_world

    #     raise ValueError("Could not find a goal position within the specified number of attempts")

    # def random_start_and_goal_path(self, entire_path: bool = False) -> Tuple[List[np.ndarray], float]:
    #     """Find a path with a random start and goal position at the average shortest path length

    #     Args:
    #         entire_path
    #             If True, return the entire path. If False, return only waypoints

    #     Returns:
    #         List[np.ndarray]
    #             List of waypoints in world coordinates
    #         float
    #             Length of the path
    #     """

    #     random_start = np.random.randint(0, self.map_size, size=2)
    #     random_start_world = np.array(self._map_to_world(random_start))
    #     goal_at_avg_path_length = self.find_goal_at_avg_shortest_path_length(random_start_world)
    #     path_world, geodesic_dist = self.shortest_path(random_start_world, goal_at_avg_path_length, entire_path=entire_path)

    #     return path_world, geodesic_dist

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
        # print('source_world: ', source_world)
        # print('target_world: ', target_world)
        # print('self.map.shape: ', self.map.shape)
        # p1 = Position(self.map.shape)
        # p1.set_position(source_world[0], source_world[1])
        # print('p1: ', p1.get_position_pix())

        # p2 = Position(self.map.shape)
        # p2.set_position(target_world[0], target_world[1])
        # print('p2: ', p2.get_position_pix())

        source = tuple(self._world_to_map(source_world))
        target = tuple(self._world_to_map(target_world))
        # source = (int(source[1]), int(source[0]))
        # target = (int(target[1]), int(target[0]))
        # print('source_map: ', source)
        # print('target_map: ', target)

        if not self.graph.has_node(source):
            # print('source not in graph')
            nodes = np.array(self.graph.nodes)
            closest_node = tuple(
                nodes[np.argmin(np.linalg.norm(nodes - source, axis=1))])
            self.graph.add_edge(closest_node, source,
                        weight=l2_distance(source, closest_node))

        if not self.graph.has_node(target):
            print('target not in graph')
            nodes = np.array(self.graph.nodes)
            closest_node = tuple(
                nodes[np.argmin(np.linalg.norm(nodes - target, axis=1))])
            self.graph.add_edge(closest_node, target, 
                        weight=l2_distance(target, closest_node))

        path_map = np.asarray(nx.astar_path(
            self.graph, source, target, heuristic=l2_distance))

        path_world = self._map_to_world(path_map)
        all_dists = np.linalg.norm(path_world[1:] - path_world[:-1], axis=1)
        geodesic_dist = np.sum(all_dists)


        if entire_path:
            waypoints = []

            # Make path coarse by getting a waypoint every half meter
            # num_points = geodesic_dist // self.waypoint_dist
            # self.num_waypoints = int(num_points) + 1
            # last_waypoint = path_world[0]
            # waypoints.append(last_waypoint)
            dist = 0

            for i in range(1, path_world.shape[0]):
                dist += all_dists[i-1]
                if dist >= self.waypoint_dist:
                    dist = 0
                    waypoints.append(path_world[i])

            waypoints.append(path_world[-1])
            waypoints = waypoints[:self.num_waypoints]
            waypoints = self.shortest_path_to_waypoints(waypoints)
            

            return np.array(waypoints), geodesic_dist


        if not entire_path:

            waypoints = []

            dist = 0

            for i in range(1, path_world.shape[0]):
                dist += all_dists[i-1]
                if dist >= self.waypoint_dist:
                    dist = 0
                    waypoints.append(path_world[i])

            path_world = np.array(waypoints[:self.num_waypoints])
            num_remaining_waypoints = self.num_waypoints - path_world.shape[0]

            if num_remaining_waypoints > 0:
                remaining_waypoints = np.tile(
                    target_world, (num_remaining_waypoints, 1))
                path_world = np.concatenate(
                    (path_world, remaining_waypoints), axis=0)

        path_world = self.shortest_path_to_waypoints(path_world)

        return np.array(path_world), geodesic_dist        

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
    
