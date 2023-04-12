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

    def process_obstacles(self)->None:
        """Process the obstacles in the environment and store them in the
        obstacles attribute
        """
        _, reachable_trav_map = self.build_graph(self.map)
        self._obstacles_polygons = self._get_obstacle_polygons(reachable_trav_map)
        self.get_obstacles()
        self._construct_obstacle_map()

    def compute_trav_map(self)->None:
        """Compute the traversability map for the environment
        """
        kernel = np.ones((indoorca.max_agent_radius, indoorca.max_agent_radius)).astype(np.uint8)
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

        return polygons
    
    def _new_construct_obstacle_map(self)->None:
        # Initialize an empty binary map
        height = self.map.shape[0]
        width = self.map.shape[1]

        #Initialize map as all freespace
        self.obstacle_map = np.ones((height, width)) * indoorca.free_space

        # Draw the obstacles on the traversability map
        for obstacle in self.obstacles_meters:
            # Convert the obstacle vertices from meters to pixels and shift the origin
            obstacle_pixels = [(int((x * self.pix_per_meter) + width // 2), int(height - ((y * self.pix_per_meter) + height // 2))) for x, y in obstacle]

            # Create a polygon object for the obstacle
            obstacle_polygon = Polygon(obstacle_pixels)

            # Create a bounding box around the obstacle
            bbox = Bbox.from_bounds(obstacle_polygon.get_extents().xmin, obstacle_polygon.get_extents().ymin,
                                    obstacle_polygon.get_extents().width, obstacle_polygon.get_extents().height)

            # Generate a grid of points inside the bounding box
            y, x = np.mgrid[int(bbox.ymin):int(bbox.ymax), int(bbox.xmin):int(bbox.xmax)]
            points = np.vstack((x.flatten(), y.flatten())).T

            # Check if the points are inside the obstacle polygon
            is_inside = np.array([obstacle_polygon.contains_point(point) for point in points]).reshape(y.shape)

            # Set the points inside the obstacle polygon to 0 (non-traversable)
            traversability_map[int(bbox.ymin):int(bbox.ymax), int(bbox.xmin):int(bbox.xmax)][is_inside] = 0


    def _construct_obstacle_map(self)->None:
        """Construct the obstacle map from the polygons
        """
        # Initialize an empty binary map
        height = self.map.shape[0]
        width = self.map.shape[1]

        #Initialize map as all freespace
        self.obstacle_map = np.ones((height, width)) * indoorca.free_space

        # for obstacle in self.obstacles:
        #     # Convert the obstacle vertices from meters to pixels
        #     pixel_coords = [list(self._world_to_map(np.asarray(coord))) for coord in obstacle]

        #     obstacle_polygon = mplPolygon(pixel_coords)

        #     # Create a bounding box around the obstacle
        #     bbox = Bbox.from_bounds(obstacle_polygon.get_extents().xmin, obstacle_polygon.get_extents().ymin,
        #                                 obstacle_polygon.get_extents().width, obstacle_polygon.get_extents().height)

        #     # Generate a grid of points inside the bounding box
        #     x, y = np.mgrid[int(bbox.xmin):int(bbox.xmax), int(bbox.ymin):int(bbox.ymax)]
        #     points = np.vstack((x.flatten(), y.flatten())).T

        #     # Check if the points are inside the obstacle polygon
        #     is_inside = np.array([obstacle_polygon.contains_point(point) for point in points]).reshape(x.shape)

        #     # Set the points inside the obstacle polygon to 0 (non-traversable)
        #     self.obstacle_map[int(bbox.ymin):int(bbox.ymax), int(bbox.xmin):int(bbox.xmax)][is_inside] = indoorca.obstacle_space



        # Iterate through each polygon
        for obstacle in self.obstacles_meters:
            # Convert the meter coordinates to pixels
            pixel_coords = np.array([self._world_to_map(np.asarray(coord)) for coord in obstacle], dtype=np.int32)

            print(pixel_coords)
            # Fill the obstacle polygon with 0 (non-traversable)
            cv2.fillPoly(self.obstacle_map, pixel_coords, indoorca.obstacle_space)
            # Account for origin being at center of image

            # pixel_coords = [(coord[0] * (width - 1) / 2, coord[1] * (height - 1) / 2) for coord in pixel_coords]

            # Create a Shapely polygon from the pixel coordinates
            # pixel_polygon = mplPolygon(pixel_coords)

            # # Iterate through all the pixels in the binary map
            # for i in range(height):
            #     for j in range(width):
            #         # Scale the pixel coordinates to the range (-1, 1)
            #         # x = (((j * 2) + width)/ (width - 1)) - 1
            #         # y = (((i * 2) + height) / (height - 1)) - 1

            #         # Check if the point is inside the polygon
            #         if pixel_polygon.contains(Point(i, j)):
            #             self.obstacle_map[i, j] = indoorca.obstacle_space


    
    def get_obstacles(self) -> None:
        """ Return a list of obstacles for the simulator.
        
        Returns
        -------
        List[List[Tuple[float]]] :
            List of obstacles' vertices for the simulator
        """
        converted_polygons = []


        for polygon in self._obstacles_polygons:
            # Convert coordinates to meters
            meter_coords = [(coord[0] * self.pix_per_meter / 2, coord[1] * self.pix_per_meter / 2) for coord in polygon.exterior.coords]

            # Append the list of tuples (coordinates) to the converted_polygons list
            converted_polygons.append(meter_coords)

        self.obstacles_meters = converted_polygons
    

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
        ret = np.flip(xy*indoorca.pix_per_meter + 
                        np.array(self.map.shape)/2.0).astype(int)
    
        #check if any of the points are negative or greater than the map size, set them to 0 or the max size if necessary
        ret[ret < 0] = 0

        ret[ret >= self.map.shape[0]] = self.map.shape[0] - 1

        return ret


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
    
