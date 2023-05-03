# import logging
# from typing import List, Tuple

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as mplPolygon
# from skimage import measure
# from shapely.geometry import Polygon, MultiPolygon, Point

# from indoorca import pix_per_meter


# class MapProcessor:
#     """ Class for processing a binary traversability map and generating obstcle
#     coordinates for the simulator.
#     """
#     def __init__(self, map: np.ndarray):
#         """ Initialize the MapProcessor.
        
#         Parameters
#         ----------
#         map :
#             Binary traversability map
#         pix_per_meter :
#             Number of pixels per meter

#         """
#         self._map = self._crop_map(map)
#         self.pix_per_meter = pix_per_meter
#         self._obstacle_polygons = self._get_obstacle_polygons(map)
#         self.obstacles = None


#     def _crop_map(self, map: np.ndarray, tol: int=0) -> np.ndarray:
#         """ Return the cropped map.
        
#         Parameters
#         ----------
#         map :
#             Binary 2D map
#         tol :
#             Tolerance. If map is equal to tol, it will be cropped. Default is 0.
        
#         Returns
#         -------
#         np.ndarray :
#             Cropped binary traversability map
#         """
#         # Find the min and max index of the non-zero elements
#         mask = map > tol
#         return map[np.ix_(mask.any(1), mask.any(0))]
    
#     def _simplify_polygon(self, polygon: Polygon, tolerance: float=0.02) -> List[Polygon]:
#         """ Return the simplified polygon.
        
#         Parameters
#         ----------
#         polygon :
#             Polygon to be simplified
#         tolerance :
#             Tolerance. Default is 0.0.
        
#         Returns
#         -------
#         Polygon :
#             Simplified polygon
#         """
#         simplified = polygon.simplify(0.02, preserve_topology=True)
#         if isinstance(simplified, Polygon):
#             return [simplified]
#         return list(simplified)
    
#     def _get_obstacle_polygons(self, 
#                                tolerance: float=0.02, 
#                                min_area_ratio: float=0.0001 
#                                )-> List[Polygon]:
#         """Process the map and return a list of Polygon objects.

#         Parameters
#         ----------
#         tolerance : optional
#             Tolerance for simplifying the polygons, by default 0.02
#         min_area_ratio : optional
#             The threshold for ignoring obstacles, by default 0.0001

#         Returns
#         -------
#         List[Polygon]
#             List of Polygon objects extracted from the map
#         """        

#         contours = measure.find_contours(self._map, 0.5)

#         polygons = []
#         for contour in contours:
#             # Convert coordinates back to the original scale
#             scaled_contour = [((point[1] * 2) / (self._map.shape[1] - 1)) - 1 for point in contour]
#             scaled_contour_y = [((point[0] * 2) / (self._map.shape[0] - 1)) - 1 for point in contour]

#             # Ensure the contour has at least 3 points to form a valid polygon
#             if len(contour) < 3:
#                 continue

#             # Create a Shapely polygon from the contour
#             polygon_coords = list(zip(scaled_contour, scaled_contour_y))
#             polygon = Polygon(polygon_coords)

#             # Discard very small polygons based on the min_area_ratio
#             if polygon.area < min_area_ratio:
#                 continue

#             # Simplify the polygon and add it to the list
#             simplified_polygons = self._simplify_polygon(polygon, tolerance)
#             polygons.extend(simplified_polygons)

#         return polygons
    
#     # def get_obstacles(self, polygons: List[Polygon]) -> List[List[float]]:
#     #     """ Return a list of obstacles for the simulator.
        
#     #     Parameters
#     #     ----------
#     #     polygons :
#     #         List of Polygon objects
        
#     #     Returns
#     #     -------
#     #     List[List[float]] :
#     #         List of obstacles' vertices for the simulator
#     #     """
#     #     converted_polygons = []

#     #     for polygon in polygons:
#     #         # Convert coordinates to meters
#     #         meter_coords = [(coord[0] * self.pix_per_meter / 2, coord[1] * self.pix_per_meter / 2) for coord in polygon.exterior.coords]

#     #         # Append the list of tuples (coordinates) to the converted_polygons list
#     #         converted_polygons.append(meter_coords)

#     #     return converted_polygons
    
#     def get_obstacles(self, pix_per_meter: int=10) -> List[List[Tuple[float]]]:
#         """ Return a list of obstacles for the simulator.

#         Parameters
#         ----------
#         pixel_per_meter :
#             Number of pixels per meter
        
#         Returns
#         -------
#         List[List[Tuple[float]]] :
#             List of obstacles' vertices for the simulator
#         """
#         converted_polygons = []

#         for polygon in self._obstacle_polygons:
#             # Convert coordinates to meters
#             meter_coords = [(coord[0] * pix_per_meter / 2, coord[1] * pix_per_meter / 2) for coord in polygon.exterior.coords]

#             # Append the list of tuples (coordinates) to the converted_polygons list
#             converted_polygons.append(meter_coords)

#         return converted_polygons
    
#     from matplotlib.patches import Polygon as mplPolygon


#     def display_map_with_polygons(self, polygon_color: str='red')-> None:
#         """ Display the binary map with the polygons.

#         Parameters
#         ----------
#         polygon_color : optional
#             Color of the polygons, by default 'red'
#         """

#         _, ax = plt.subplots()

#         # Display the binary map
#         ax.imshow(self._map, cmap='gray', extent=(-1, 1, -1, 1), origin='lower')

#         # Plot the polygons
#         for polygon in self._obstacle_polygons:
#             coords = list(polygon.exterior.coords)
#             coords_tuples = [(x, y) for x, y in coords]

#             # Ensure the coordinates have at least 3 points to form a valid polygon
#             if len(coords_tuples) < 3:
#                 continue

#             mpl_polygon = mplPolygon(coords_tuples, edgecolor=polygon_color, facecolor='none', linewidth=2)
#             ax.add_patch(mpl_polygon)

#         ax.set_xlim(-1, 1)
#         ax.set_ylim(-1, 1)
#         ax.set_aspect('equal', 'box')
#         plt.show()  
