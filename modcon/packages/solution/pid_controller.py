from typing import Tuple
import numpy as np
import yaml
import os

def PIDController(
    v_0: float,
    theta_ref: float,
    theta_hat: float,
    prev_e: float,
    prev_int: float,
    delta_t: float
) -> Tuple[float, float, float, float]:
    """
    PID performing heading control.
    Args:
        v_0: linear Duckiebot speed (given).
        theta_ref: reference heading pose.
        theta_hat: the current estiamted theta.
        prev_e: tracking error at previous iteration.
        prev_int: previous integral error term.
        delta_t: time interval since last call.
    Returns:
        v_0: linear velocity of the Duckiebot
        omega: angular velocity of the Duckiebot
        e: current tracking error (automatically becomes prev_e at next iteration).
        e_int: current integral error (automatically becomes prev_int at next iteration).
    """
    # Read PID gains from file
    script_dir = os.path.dirname(__file__)
    file_path = script_dir + "/GAINS.yaml"
    try:
        with open(file_path) as f:
            gains = yaml.full_load(f)
        Kp = gains['kp']
        Ki = gains['ki']
        Kd = gains['kd']
    except (FileNotFoundError, KeyError):
        # Default values if file not found or missing keys
        print(f"Warning: Could not load gains from {file_path}. Using default values.")
        Kp = 5.0
        Ki = 0.2
        Kd = 0.1
    
    # Tracking error
    e = theta_ref - theta_hat
    
    # Integral of the error
    e_int = prev_int + e * delta_t
    
    # Anti-windup - preventing the integral error from growing too much
    e_int = max(min(e_int, 2), -2)
    
    # Derivative of the error
    e_der = (e - prev_e) / delta_t if delta_t > 0 else 0
    
    # PID controller for omega
    omega = Kp * e + Ki * e_int + Kd * e_der
    
    # Ensure omega is not too small to cause movement
    if abs(omega) < 0.1 and abs(e) > 0.01:
        # If there's a non-trivial error but omega is very small, enforce a minimum
        omega = 0.5 if e > 0 else -0.5
    
    # Debug output
    print("===== DEBUG INFORMATION =====")
    print(f"v_0: {v_0}")
    print(f"theta_ref: {np.rad2deg(theta_ref):.2f} deg")
    print(f"theta_hat: {np.rad2deg(theta_hat):.2f} deg")
    print(f"error: {np.rad2deg(e):.2f} deg")
    print(f"Kp: {Kp}, Ki: {Ki}, Kd: {Kd}")
    print(f"P term: {Kp*e:.2f}, I term: {Ki*e_int:.2f}, D term: {Kd*e_der:.2f}")
    print(f"omega output: {omega:.2f}")
    print("=============================")
    
    # Ensure v_0 is positive
    v_0 = max(v_0, 0.05)  # Minimum velocity of 0.05
    
    return v_0, omega, e, e_int