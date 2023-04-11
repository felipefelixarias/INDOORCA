from typing import List

import numpy as np

def l2_distance(v1: List[float], v2: List[float]) -> float:
    """Get the distance between two vectors

    Parameters
    ----------
    v1
        First vector
    v2
        Second vector

    Returns
    -------
    float
        Distance between the two vectors
    """
    return np.linalg.norm(np.array(v1) - np.array(v2))