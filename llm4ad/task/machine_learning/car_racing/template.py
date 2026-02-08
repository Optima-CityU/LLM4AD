template_program = '''
import numpy as np
import cv2
    
def choose_action(observation, car_speed, pre_action, pre_observation):
    """
    Determine the next action for the Car Racing agent.
    This function takes into account the current state (observation and speed), the previous action, and the previous observation.

    Notes:
    - The car in this environment is a powerful rear-wheel-drive vehicle. Avoid accelerating while turning sharply,
      as this can easily lead to loss of control.
    - Occasionally, track segments (e.g., after a U-turn) may appear in the observation but are not part of the immediate drivable path. These should be distinguished to avoid premature or incorrect decisions.
    - Avoid coming to a complete stop, as this may prevent the car from finishing the race.

    Args:
        observation (np.ndarray): The current state observed by the agent, represented as a 96x96 RGB image
                                   of the car and race track from a top-down view (shape: (96, 96, 3)).

        car_speed (float): The current speed of the car.

        pre_action (np.ndarray): The action taken by the agent in the previous step, represented as a
                                  3-element array.

        pre_observation (np.ndarray): The observation received when the previous action was taken. It has the same shape and format as `observation` (i.e., a 96x96 RGB image).
        

    Returns:
        np.ndarray: The action selected by the agent for the next step, represented as an array of shape (3,) where:
                    - Index 0: Steering, where -1 is full left, +1 is full right (range: [-1, 1]).
                    - Index 1: Gas, (range: [0, 1]).
                    - Index 2: Braking, (range: [0, 1]).
    """
    action = np.array([0.0, 0.0, 0.0])
    # Gray track detection parameters (RGB 95-115 range with ±5% tolerance)
    gray_low = 95
    gray_high = 115

    # Create 3D gray detection mask (all RGB channels within range)
    gray_mask = (
            (observation[:, :, 0] >= gray_low) & (observation[:, :, 0] <= gray_high) &
            (observation[:, :, 1] >= gray_low) & (observation[:, :, 1] <= gray_high) &
            (observation[:, :, 2] >= gray_low) & (observation[:, :, 2] <= gray_high)
    )

    gray_indices = np.argwhere(gray_mask)
    center_x = np.mean(gray_indices[:, 1]) if len(gray_indices) > 0 else observation.shape[1] // 2
    car_position = observation.shape[1] // 2
    offset = center_x - car_position

    steering_angle = np.clip(offset / 100.0, -1.0, 1.0)
    action[0] = steering_angle

    if abs(offset) > 10:
        action[1] = 0.0
        action[2] = 0.2
    else:
        action[1] = 0.8
        action[2] = 0.0

    gray_density = np.sum(gray_mask) / (observation.shape[0] * observation.shape[1])
    if gray_density < 0.1:
        action[1] = 0.4
        action[2] = 0.3

    return action
'''

task_description = (
    "Write a Python function that serves as a control strategy for an agent in a top-down car racing environment. \n\n"
    "### Environment Overview\n"
    "In this environment, the agent is required to drive a car along a race track. The primary objective is to cover as much of the track surface as possible before the time limit expires. To accomplish this, the agent needs to efficiently navigate the track by controlling the car's steering, throttle, and brake.\n\n"
    "### Observation Details\n"
    "The agent's observation consists of a 96x96x3 RGB image representing the top-down view of the environment. The following key visual elements can be identified in this image:\n"
    "- Car: Red (approximately [~202, <10, <10]).\n"
    "- Track: Gray (approximately [~102, ~102, ~102]).\n"
    "- Off-track grass: Greenish (approximately [~102, ~204, ~102]).\n"
    "- Curbs (Sharp Turns): High-contrast red and white (approximately [>240, <20, <20] and [>240, >240, >240]).\n\n"
    "### Inputs at Each Time Step\n"
    "At every time step, the agent receives the following information:\n"
    "- The current RGB observation of the environment.\n"
    "- The current speed of the car.\n"
    "- The previous RGB observation and the previous action taken by the agent.\n\n"
    "### Function Requirements\n"
    "The Python function should incorporate a control policy that combines visual perception from the RGB observations and past information (previous observation and action). This policy should enable the agent to maintain optimal control of the car, keep the car on the track, and maximize the efficiency of lap completion."
)

