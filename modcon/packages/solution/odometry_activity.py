from typing import Tuple
import numpy as np

def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    """
        Args:
            ticks: Current tick count from the encoders.
            prev_ticks: Previous tick count from the encoders.
            resolution: Number of ticks per full wheel rotation returned by the encoder.
        Return:
            dphi: Rotation of the wheel in radians.
            ticks: current number of ticks.
    """
    # Calculate difference in tick counts
    delta_ticks = ticks - prev_ticks
    
    # Convert tick difference to radians
    # Full wheel rotation (2π radians) divided by resolution and multiplied by ticks
    dphi = 2 * np.pi * delta_ticks / resolution
    
    return dphi, ticks

def pose_estimation(
        R: float,
        baseline: float,
        x_prev: float,
        y_prev: float,
        theta_prev: float,
        delta_phi_left: float,
        delta_phi_right: float,
) -> Tuple[float, float, float]:
    """
    Calculate the current Duckiebot pose using the dead-reckoning model.
    Args:
        R: radius of wheel (both wheels are assumed to have the same size) - this is fixed in simulation,
        and will be imported from your saved calibration for the real robot
        baseline: distance from wheel to wheel; 2L of the theory
        x_prev: previous x estimate - assume given
        y_prev: previous y estimate - assume given
        theta_prev: previous orientation estimate - assume given
        delta_phi_left: left wheel rotation (rad)
        delta_phi_right: right wheel rotation (rad)
    Return:
        x_curr: estimated x coordinate
        y_curr: estimated y coordinate
        theta_curr: estimated heading
    """
    # Calculate linear distance traveled by each wheel
    d_left = R * delta_phi_left
    d_right = R * delta_phi_right
    
    # Calculate forward distance traveled by robot (average of both wheels)
    d_center = (d_left + d_right) / 2
    
    # Calculate change in orientation
    delta_theta = (d_right - d_left) / baseline
    
    # Update position based on current heading
    x_curr = x_prev + d_center * np.cos(theta_prev)
    y_curr = y_prev + d_center * np.sin(theta_prev)
    
    # Update heading
    theta_curr = theta_prev + delta_theta
    
    return x_curr, y_curr, theta_curr