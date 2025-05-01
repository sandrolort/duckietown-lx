from typing import Tuple
import numpy as np

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Creates a matrix for the left motor activation that responds to obstacles
    on the left side of the image (so the robot turns right away from them).
    
    Args:
        shape: Tuple of (height, width) dimensions
        
    Returns:
        A numpy array with weights for left motor activation
    """
    height, width = shape
    res = np.zeros(shape=shape, dtype="float32")
    
    # Create a pattern that activates the left motor when objects are on the left
    for y in range(height):
        for x in range(width):
            # Calculate vertical position factor (lower = more important)
            y_factor = y / height  # 0.0 at top, 1.0 at bottom
            y_weight = np.power(y_factor, 2)  # Emphasize lower portion
            
            # Calculate horizontal position (left side of image affects left motor)
            x_factor = 1.0 - (x / width)  # 1.0 at left, 0.0 at right
            
            # Set negative weights on the left side (to slow down left motor when seeing objects)
            weight = -y_weight * x_factor if x < width/2 else 0
            res[y, x] = -weight
            
    # Normalize to make sure min value is -1.0
    if np.min(res) < 0:
        res = res / abs(np.min(res))
                
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """
    Creates a matrix for the right motor activation that responds to obstacles
    on the right side of the image (so the robot turns left away from them).
    
    Args:
        shape: Tuple of (height, width) dimensions
        
    Returns:
        A numpy array with weights for right motor activation
    """
    height, width = shape
    res = np.zeros(shape=shape, dtype="float32")
    
    # Create a pattern that activates the right motor when objects are on the right
    for y in range(height):
        for x in range(width):
            # Calculate vertical position factor (lower = more important)
            y_factor = y / height  # 0.0 at top, 1.0 at bottom
            y_weight = np.power(y_factor, 2)  # Emphasize lower portion
            
            # Calculate horizontal position (right side of image affects right motor)
            x_factor = x / width  # 0.0 at left, 1.0 at right
            
            # Set negative weights on the right side (to slow down right motor when seeing objects)
            weight = -y_weight * x_factor if x >= width/2 else 0
            res[y, x] = -weight
            
    # Normalize to make sure min value is -1.0
    if np.min(res) < 0:
        res = res / abs(np.min(res))
                
    return res