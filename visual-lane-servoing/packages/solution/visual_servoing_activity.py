from typing import Tuple
import numpy as np
import cv2

def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.
    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """
    h, w = shape
    steer_matrix_left = np.zeros((h, w), dtype=np.float32)

    # For left (yellow) lane markings:
    # - Yellow on the left side should make us steer RIGHT (positive values)
    # - Make the gradient steeper for more pronounced turning
    for i in range(h):
        for j in range(w):
            # Use non-linear mapping for stronger steering response
            normalized_pos = j / w  # 0 on left, 1 on right
            # Apply exponential amplification - stronger turning effect
            if normalized_pos < 0.5:
                # Left side of image - steer RIGHT (positive values)
                steer_matrix_left[i, j] = -1.5  # Strong positive steering
            else:
                # Right side of image - steer RIGHT less aggressively
                steer_matrix_left[i, j] = -0.5
            
    return steer_matrix_left

def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.
    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """
    h, w = shape
    steer_matrix_right = np.zeros((h, w), dtype=np.float32)
    
    # For right (white) lane markings:
    # - White on the right side should make us steer LEFT (negative values)
    # - Make the gradient steeper for more pronounced turning
    for i in range(h):
        for j in range(w):
            # Use non-linear mapping for stronger steering response
            normalized_pos = j / w  # 0 on left, 1 on right
            # Apply exponential amplification - stronger turning effect
            if normalized_pos > 0.5:
                # Right side of image - steer LEFT (negative values)
                steer_matrix_right[i, j] = 1.5  # Strong negative steering
            else:
                # Left side of image - steer LEFT less aggressively
                steer_matrix_right[i, j] = 0.8
            
    return steer_matrix_right

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    # Convert to HSV and grayscale
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create ground mask with horizon cutoff
    mask_ground = np.ones(img.shape[:2], dtype=np.uint8)
    horizon_line = int(img.shape[0] * 3/7)
    mask_ground[:horizon_line, :] = 0
    
    # Apply Gaussian blur
    sigma = 3
    img_gaussian_filter = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # Compute Sobel derivatives for edge enhancement
    sobelx = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img_gaussian_filter, cv2.CV_64F, 0, 1)
    
    # Compute gradient magnitude for edge detection
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    threshold = 60
    mask_mag = (Gmag > threshold).astype(np.uint8)
    
    # Define color thresholds for white and yellow lane markings
    white_lower_hsv = np.array([0, 0, 170])
    white_upper_hsv = np.array([180, 45, 255])
    yellow_lower_hsv = np.array([15, 80, 150])
    yellow_upper_hsv = np.array([35, 255, 255])
    
    # Create color masks
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
    
    # Create left and right half masks
    width = img.shape[1]
    mask_left = np.ones(img.shape[:2])
    mask_left[:, int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(img.shape[:2])
    mask_right[:, 0:int(np.floor(width/2))] = 0
    
    # Create the final masks - incorporate edge detection for better line detection
    mask_left_edge = mask_ground * mask_left * mask_yellow * mask_mag
    mask_right_edge = mask_ground * mask_right * mask_white * mask_mag
    
    # Weight masks by vertical position (closer to car = more important)
    h = img.shape[0]
    for i in range(h):
        # Weight increases linearly from 0.5 at horizon to 1.0 at bottom
        weight = 0.5 + 0.5 * (i - horizon_line) / (h - horizon_line)
        if i >= horizon_line:  # Only apply weighting below horizon
            mask_left_edge[i, :] *= weight
            mask_right_edge[i, :] *= weight
    
    return mask_left_edge, mask_right_edge

# Additional functions for speed control:

def get_speed_value(left_mask: np.ndarray, right_mask: np.ndarray) -> float:
    """
    Calculate appropriate speed based on lane detection
    
    Args:
        left_mask: Masked image for the left lane marking
        right_mask: Masked image for the right lane marking
        
    Return:
        speed: Speed value (0.0 to 1.0)
    """
    # Default base speed (when lanes are well detected)
    default_speed = 0.5  # 50% of maximum speed
    
    # Calculate how much of each lane is detected
    left_detection = np.sum(left_mask) / left_mask.size
    right_detection = np.sum(right_mask) / right_mask.size
    
    # Calculate total lane detection confidence
    total_detection = left_detection + right_detection
    
    # If we detect good lane markings, maintain default speed
    # If detection is poor, reduce speed proportionally
    confidence_threshold = 0.01  # Minimum detection needed for full speed
    
    if total_detection < confidence_threshold:
        # Scale speed based on detection quality
        speed = default_speed * (total_detection / confidence_threshold)
        # Never go below minimum speed
        speed = max(speed, 0.1)
    else:
        speed = default_speed
    
    return speed

# Example of how to use these functions in a main control loop:
def lane_following_control(image: np.ndarray):
    """
    Main control function that processes an image and returns steering and speed
    
    Args:
        image: Camera image in BGR format
        
    Return:
        steering: Steering value (-1.0 to 1.0)
        speed: Speed value (0.0 to 1.0)
    """
    # Detect lane markings
    left_mask, right_mask = detect_lane_markings(image)
    
    # Get steering matrices
    h, w = image.shape[:2]
    left_steer_matrix = get_steer_matrix_left_lane_markings((h, w))
    right_steer_matrix = get_steer_matrix_right_lane_markings((h, w))
    
    # Calculate steering based on detected lanes
    left_steering = np.sum(left_mask * left_steer_matrix) / (np.sum(left_mask) + 1e-6)
    right_steering = np.sum(right_mask * right_steer_matrix) / (np.sum(right_mask) + 1e-6)
    
    # Combine steering inputs with appropriate weights
    # Adjust these weights to balance the influence of each lane
    left_weight = 0.6z
    right_weight = 0.2
    steering = (left_weight * left_steering + right_weight * right_steering) 
    
    # Limit steering to valid range
    steering = np.clip(steering, -1.0, 1.0)
    
    # Calculate speed based on lane detection
    speed = get_speed_value(left_mask, right_mask)
    
    return steering, speed