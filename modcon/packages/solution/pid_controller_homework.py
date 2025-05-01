from typing import Tuple
import numpy as np
import yaml
import os

def PIDController(
    v_0: float, y_ref: float, y_hat: float, prev_e_y: float, prev_int_y: float, delta_t: float
) -> Tuple[float, float, float, float]:
    """
    PID performing lateral control.
    Args:
        v_0: linear Duckiebot speed (constant).
        y_ref: target y coordinate.
        y_hat: the current estimated y.
        prev_e_y: tracking error at previous iteration.
        prev_int_y: previous integral error term.
        delta_t: time interval since last call.
    Returns:
        v_0: linear velocity of the Duckiebot
        omega: angular velocity of the Duckiebot
        e: current tracking error (automatically becomes prev_e_y at next iteration).
        e_int: current integral error (automatically becomes prev_int_y at next iteration).
    """
    # Read PID gains from file
    script_dir = os.path.dirname(__file__)
    file_path = script_dir + "/GAINS.yaml"
    try:
        with open(file_path) as f:
            gains = yaml.full_load(f)
        kp = gains['kp']
        ki = gains['ki']
        kd = gains['kd']
    except (FileNotFoundError, KeyError):
        # Default values if file not found or missing keys
        print(f"Warning: Could not load gains from {file_path}. Using default values.")
        kp = 6.0
        ki = 0.5
        kd = 0.3
    
    # Calculate error terms
    e = y_ref - y_hat  # Lateral error
    
    # Integral error term with anti-windup
    e_int = prev_int_y + e * delta_t
    e_int = max(min(e_int, 2), -2)  # Limit integral term to prevent windup
    
    # Derivative error term
    e_der = (e - prev_e_y) / delta_t if delta_t > 0 else 0
    
    # Calculate omega using PID formula
    omega = kp * e + ki * e_int + kd * e_der
    
    # Ensure omega is not too small to cause movement
    if abs(omega) < 0.1 and abs(e) > 0.01:
        # If there's a non-trivial error but omega is very small, enforce a minimum
        omega = 0.5 if e > 0 else -0.5
    
    # Debug output
    print("===== DEBUG INFORMATION =====")
    print(f"v_0: {v_0}")
    print(f"y_ref: {y_ref:.4f}")
    print(f"y_hat: {y_hat:.4f}")
    print(f"error: {e:.4f}")
    print(f"kp: {kp}, ki: {ki}, kd: {kd}")
    print(f"P term: {kp*e:.4f}, I term: {ki*e_int:.4f}, D term: {kd*e_der:.4f}")
    print(f"omega output: {omega:.4f}")
    print("=============================")
    
    # Ensure v_0 is positive
    v_0 = max(v_0, 0.05)  # Minimum velocity of 0.05
    
    return v_0, omega, e, e_int