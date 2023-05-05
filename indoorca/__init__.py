# from numpy import rint
from math import ceil
# from opam.utils.annotation import make_gaussian

#TODO move make_gaussian to another utils
#A function that is used by multiple classes should be defined in 
#opam.utils, if it is only used for a specific class it should be 
#defined in that class' utils file

__version__ = "0.1.0"


#Global variables in meters
pix_per_meter = 100
agent_radius = 1.5
max_agent_radius = ceil(agent_radius)*pix_per_meter//10
min_agent_radius = round(agent_radius)*pix_per_meter//10
agent_diameter = max_agent_radius + min_agent_radius
# agent_mask = rint(make_gaussian(agent_diameter, fwhm=agent_diameter))

#Other global variables
obstacle_space = 0
free_space = 1
radius_meters = 0.125
radius_pixels = round(radius_meters*pix_per_meter)
