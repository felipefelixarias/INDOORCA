from typing import Tuple

import indoorca

class Position():
    def __init__(self, map_shape: Tuple[int, int]):
        self.x = 0
        self.y = 0
        self.col = 0
        self.row = 0
        self.col_lims = (0, map_shape[1]-1)
        self.row_lims = (0, map_shape[0]-1)
        self.half_map_width = map_shape[1] / 2
        self.x_lims = (-self.half_map_width / indoorca.pix_per_meter, 
                        self.half_map_width / indoorca.pix_per_meter)
        self.half_map_height = map_shape[0] / 2
        self.y_lims = (-self.half_map_height / indoorca.pix_per_meter,
                        self.half_map_height / indoorca.pix_per_meter)
        
        # print('x_lims: ', self.x_lims)
        # print('y_lims: ', self.y_lims)
        # print('col_lims: ', self.col_lims)
        # print('row_lims: ', self.row_lims)


    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def get_position_pix(self) -> Tuple[int, int]:
        return (self.row, self.col)

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
        self.row, self.col = self.meters_to_pix(x, y)

    def set_position_pix(self, row: int, col: int) -> None:
        #Make sure the position is within the map
        if col < self.col_lims[0]:
            raise ValueError('x position is less than the minimum x position of the map.')
        elif col > self.col_lims[1]:
            raise ValueError('x position is greater than the maximum x position of the map.')
        elif row < self.row_lims[0]:
            raise ValueError('y position is less than the minimum y position of the map.')
        elif row > self.row_lims[1]:
            raise ValueError('y position is greater than the maximum y position of the map.')
        
        self.col = col
        self.row = row
        self.x, self.y = self.pix_to_meters(row, col)

    def meters_to_pix(self, x: float, y: float) -> Tuple[int, int]:

        col = int(x * indoorca.pix_per_meter+self.half_map_width)
        row = int(y* indoorca.pix_per_meter+self.half_map_height)

        #Make sure the position is within the map
        if col < self.col_lims[0]:
            raise ValueError('x position is less than the minimum x position of the map.')
        elif col > self.col_lims[1]:
            raise ValueError('x position is greater than the maximum x position of the map.')
        elif row < self.row_lims[0]:
            raise ValueError('y position is less than the minimum y position of the map.')
        elif row > self.row_lims[1]:
            raise ValueError('y position is greater than the maximum y position of the map.')
        
        return (row, col)
    
    def pix_to_meters(self, row: int, col: int) -> Tuple[float, float]:
        y = (row - self.half_map_height)/ indoorca.pix_per_meter
        x = (col - self.half_map_width)/ indoorca.pix_per_meter 

        # print('x in meters: ', x)
        # print('y in meters: ', y)
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