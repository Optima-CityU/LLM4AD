template_program = '''
import numpy as np

def choose_action(s: list, last_action: int, s_pre: list) -> int:
    """
    Selects an action for the Lunar Lander to achieve a safe landing at the target location (0, 0).

    Args:
        s (list or np.ndarray): The current state of the lander. Elements:
            s[0] - horizontal position (x)
            s[1] - vertical position (y)
            s[2] - horizontal velocity (v_x)
            s[3] - vertical velocity (v_y)
            s[4] - angle (radians)
            s[5] - angular velocity
            s[6] - 1 if the first leg is in contact with the ground, else 0
            s[7] - 1 if the second leg is in contact with the ground, else 0

        last_action (int): The action taken in the previous step. One of:
            0 - do nothing
            1 - fire left orientation engine
            2 - fire main (upward) engine
            3 - fire right orientation engine

        s_pre (list or np.ndarray): The state of the lander *before* the last action was executed. Elements:
            s_pre[0] - horizontal position (x) before the last action
            s_pre[1] - vertical position (y) before the last action
            s_pre[2] - horizontal velocity (v_x) before the last action
            s_pre[3] - vertical velocity (v_y) before the last action
            s_pre[4] - angle (radians) before the last action
            s_pre[5] - angular velocity before the last action
            s_pre[6] - 1 if the first leg was in contact with the ground before the last action, else 0
            s_pre[7] - 1 if the second leg was in contact with the ground before the last action, else 0

    Returns:
        int: The chosen action for the next step. One of:
            0 - do nothing
            1 - fire left orientation engine
            2 - fire main (upward) engine
            3 - fire right orientation engine
    """
    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif angle_todo < -0.05:
        a = 3
    elif angle_todo > +0.05:
        a = 1
    return a
'''

# template_program = '''
# import numpy as np
#
# def choose_action(
#     xc: float, yc: float,
#     xv: float, yv: float,
#     a: float, av: float,
#     lc: float, rc: float,
#     last_action: int,
#     prev_xc: float, prev_yc: float,
#     prev_xv: float, prev_yv: float,
#     prev_a: float, prev_av: float,
#     prev_lc: float, prev_rc: float
# ) -> int:
#     """
#     An action selection function for a lunar lander aiming to safe land at the target location (0, 0).
#
#     Args:
#         xc (float): Current x-coordinate of the lander.
#         yc (float): Current y-coordinate of the lander.
#         xv (float): Current horizontal (x-axis) velocity.
#         yv (float): Current vertical (y-axis) velocity.
#         a (float): Current rotation angle of the lander in radians.
#         av (float): Current angular velocity (rate of change of angle).
#         lc (float): 1 if the left leg is in contact with the ground, otherwise 0.
#         rc (float): 1 if the right leg is in contact with the ground, otherwise 0.
#         last_action (int): The last action taken by the lander. Should be one of:
#                            0 (do nothing), 1 (fire left orientation engine),
#                            2 (fire main engine), 3 (fire right orientation engine).
#
#         prev_xc (float): Previous x-coordinate of the lander.
#         prev_yc (float): Previous y-coordinate of the lander.
#         prev_xv (float): Previous horizontal velocity.
#         prev_yv (float): Previous vertical velocity.
#         prev_a (float): Previous rotation angle of the lander.
#         prev_av (float): Previous angular velocity.
#         prev_lc (float): 1 if the left leg was in contact with the ground previously, otherwise 0.
#         prev_rc (float): 1 if the right leg was in contact with the ground previously, otherwise 0.
#
#     Returns:
#         int: The selected action to perform. One of:
#              0 - do nothing
#              1 - fire left orientation engine
#              2 - fire main (upward) engine
#              3 - fire right orientation engine
#     """
#     action = np.random.randint(4)  # Replace with actual policy logic
#     return action
# '''

task_description = (
    "Implement a novel heuristic strategy function that guides the lander in selecting actions step-by-step "
    "to achieve a safe landing. At each step, an appropriate action could be chosen based on the lander's "
    "current state and previous state, with the objective of reaching the target location in as few steps as possible. "
    "A 'safe landing' is defined as a touchdown with low vertical speed, upright orientation, and both "
    "angular velocity and angle close to zero, and both legs in contact with the ground."
)

non_image_representation_explanation = ("The execution result is a two-dimensional list where each element is an 8-dimensional "
                                        "vector that records the state of the lander during its landing process at intervals. "
                                        "Each vector contains: the coordinates of the lander in x and y, its linear velocities "
                                        "in x and y, its angle, its angular velocity, and two booleans indicating whether each "
                                        "leg is in contact with the ground. The units of the state are as follows: 'x': (units), "
                                        "'y': (units), 'vx': (units/second), 'vy': (units/second), 'angle': (radians), 'angular velocity': (radians/second).")


