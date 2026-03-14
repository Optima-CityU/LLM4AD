# Module Name: MLES
# Last Revision: 2026/2/9
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Qinglong Hu, Xialiang Tong, Mingxuan Yuan, Fei Liu, Zhichao Lu, and Qingfu Zhang.
#       "Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies."
#       The Fourteenth International Conference on Learning Representations (ICLR). 2026.

# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Tuple, List, Any, Set
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import copy
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import time

from llm4ad.base import Evaluation
# =========================================================================
# 🛠️ USER DEFINED: Import your custom template and task description here
# =========================================================================
from llm4ad.task.machine_learning.car_racing.template import template_program, task_description


__all__ = ['RacingCarEvaluation']

class RacingCarEvaluation(Evaluation):
    """Evaluator for CarRacing/custom control problems."""

    def __init__(self, whocall='Eoh', max_steps=1200, timeout_seconds=180, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )
        self.whocall = whocall

        # =========================================================================
        # 🛠️ USER DEFINED (i): Environment Configuration
        # Modify these variables to match your specific environment parameters.
        # =========================================================================
        self.env_name = "CarRacing-v3"
        self.env_max_episode_steps = max_steps
        objective_value = kwargs.get('objective_value', 230)
        self.final_objective_score = objective_value
        self.env_mode = kwargs.get("env_mode", 'rgb_array')

        # =========================================================================
        # 🔒 BOILERPLATE - DO NOT MODIFY
        # Instance set and mode handling for the evaluation pipeline.
        # =========================================================================
        self._mode = kwargs.get('run_mode', 'Training')
        self.instance_set = kwargs.get('instance_set')
        self.instance_id_set = tuple(self.instance_set.keys())
        if self._mode == 'Training' and not self.instance_set:
            raise ValueError("Training instance set (instance_set) not provided.")

        self.ins_to_be_solve_set = kwargs.get('ins_to_be_solve_set')
        self.to_be_solve_instance_id_set = tuple(self.ins_to_be_solve_set.keys())
        if self._mode == 'Using' and not self.ins_to_be_solve_set:
            raise ValueError("Testing instance set (ins_to_be_solve_set) not provided.")

        if self._mode == 'Combined' and (not self.instance_set or not self.ins_to_be_solve_set):
            raise ValueError("Missing Training or Testing instance set.")

    def evaluate(self, action_select: callable, ins_to_be_evaluated_id: Set | List | None = None, training_mode=True) -> \
            Optional[dict]:
        """
        🔒 MOSTLY BOILERPLATE: Aggregates results across instances.
        🛠️ Users only need to modify the final return dictionary if they want to track extra custom data.
        """
        ins_to_be_evaluated_set = self.instance_set
        if not training_mode:
            ins_to_be_evaluated_set = self.ins_to_be_solve_set
        if not ins_to_be_evaluated_id:
            ins_to_be_evaluated_id = set(self.instance_set.keys())
            if not training_mode:
                ins_to_be_evaluated_id = set(self.ins_to_be_solve_set.keys())

        instance_performance = {}
        total_rewards = {}
        image64s = {}
        episodes_recorder = {}

        for ins_id in ins_to_be_evaluated_id:
            env_seed = ins_to_be_evaluated_set[ins_id]
            each_evaluate_result = self.evaluate_single(action_select, env_seed=env_seed, skip_frame=1)

            if each_evaluate_result is not None:
                infos, img_base64 = each_evaluate_result
                total_rewards[ins_id] = infos['track_coverage'] # <--- NOTE: Adjust metric here if needed
                image64s[ins_id] = img_base64
                episodes_recorder[ins_id] = infos
                instance_performance[ins_id] = {
                    'score': infos['episode_reward'],
                    'evaluate_time': infos['evaluate_time']
                }
            else:
                print(f"Warning: Instance {ins_id} returned None.")
                instance_performance[ins_id] = {'score': -float('inf'), 'evaluate_time': 0}

        if not total_rewards:
            print("Evaluation failed: No valid rewards were collected.")
            return None

        mean_reward = np.mean(list(total_rewards.values()))
        min_reward_id = min(total_rewards, key=total_rewards.get)
        chosen_image_base64 = image64s[min_reward_id]  # Default to the image with the lowest score for debugging

        sorted_keys = sorted(instance_performance.keys())
        list_performance = [instance_performance[k]['score'] for k in sorted_keys]

        # =========================================================================
        # 🛠️ USER DEFINED (iii): Final Return Formatting
        # If whocall == 'mles', you MUST return 'score' and 'image'.
        # You may add any other keys to track custom statistics.
        # =========================================================================

        if self.whocall == 'mles':
            return {
                    # REQUIRED BY MLES:
                    'score': mean_reward,
                    'image': chosen_image_base64,

                    # CUSTOM USER METRICS (Saved in population records):
                    'Test result': episodes_recorder,
                    'observation': None,
                    'all_ins_performance': instance_performance,
                    'list_performance': list_performance
                    }
        elif self.whocall == 'dyca':
            return {'all_ins_performance': instance_performance,
                    'list_performance': list_performance}  # {int ID:{'score': 0.1, 'evaluation_time':2}, ...}
        else:
            return mean_reward

    def evaluate_single(self, action_select: callable, env_seed=42, skip_frame=1):
        """
        # =========================================================================
        # 🛠️ USER DEFINED (ii): Single Episode Evaluation & Image Generation
        # This is the core logic. You must run your environment, collect rewards,
        # generate a plot/image demonstrating the behavior, and return it as base64.
        # =========================================================================
        """
        env = gym.make(self.env_name, render_mode=self.env_mode, domain_randomize=False, continuous=True)  # 'rgb_array'
        observation, _ = env.reset(seed=env_seed)  # initialization
        start_time = time.time()

        # Initialize custom variables for tracking (Specific to CarRacing example)
        action = np.array([0.0, 1.0, 0.0])
        episode_reward = 0
        episode_max_reward = 0
        trajectory = []
        car_angles = []
        view_rectangles = []
        done = False

        pre_observation = copy.deepcopy(observation)
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step = 0

        # --- RUN ENVIRONMENT LOOP ---
        while not done and step < self.env_max_episode_steps:
            # Extract state for the generated policy
            car_velocity = env.unwrapped.car.hull.linearVelocity
            speed = np.sqrt(car_velocity[0] ** 2 + car_velocity[1] ** 2)

            # Execute generated policy
            action = action_select(observation,
                                   speed,
                                   action,
                                   pre_observation)
            pre_observation = copy.deepcopy(observation)

            for _ in range(skip_frame):
                observation, reward, done, truncated, info = env.step(action)
                step += 1

                # Track data specifically needed for generating the behavior evidence plot
                car_pos = env.unwrapped.car.hull.position
                car_angle = env.unwrapped.car.hull.angle
                trajectory.append((car_pos.x, car_pos.y))
                car_angles.append(car_angle)
                corrected_angle = car_angle + np.pi / 2
                view_center_x = car_pos.x + np.cos(corrected_angle) * 14.0
                view_center_y = car_pos.y + np.sin(corrected_angle) * 14.0

                view_rectangles.append((view_center_x, view_center_y, corrected_angle, 38.0, 46.0))

                episode_reward += reward
                episode_max_reward = max(episode_max_reward, episode_reward)

        # --- GENERATE BEHAVIORAL EVIDENCE (BE) PLOT ---
        # The MLLM needs to *see* why the policy succeeded or failed.
        plt.figure(figsize=(9, 8))
        # (Plotting logic omitted for brevity in template, but keep your drawing code here! you can also directly input your image)
        # ... [Draw track, trajectories, bounding boxes, etc.] ...
        # =========================================================================
        green_color = '#62f972'
        plt.gca().set_facecolor(green_color)
        for polygon in env.unwrapped.road_poly:
            vertices = polygon[0]
            color = polygon[1]
            if hasattr(color, '__iter__') and not isinstance(color, tuple):
                color = tuple(color)
            fill_color = '#666666'  # (102,102,102)

            if isinstance(color, tuple) and len(color) == 3:
                r = max(0, min(255, int(round(color[0]))))
                g = max(0, min(255, int(round(color[1]))))
                b = max(0, min(255, int(round(color[2]))))

                fill_color = "#{:02X}{:02X}{:02X}".format(r, g, b)

            x_coords = [v[0] for v in vertices] + [vertices[0][0]]
            y_coords = [v[1] for v in vertices] + [vertices[0][1]]

            plt.fill(x_coords, y_coords, color=fill_color, alpha=1.0)
        view_color = '#8000FF'
        arrow_interval = 40
        for idy, rect in enumerate(view_rectangles):
            if idy == 0 or idy == len(view_rectangles) - 1 or idy % arrow_interval == 0:
                center_x, center_y, angle, length, width = rect

                rect_patch = patches.Rectangle(
                    (-length / 2, -width / 2),
                    length,
                    width,
                    linewidth=0,
                    edgecolor='none',
                    facecolor=view_color,
                    alpha=0.1
                )

                t = Affine2D().rotate(angle).translate(center_x, center_y) + plt.gca().transData
                rect_patch.set_transform(t)
                plt.gca().add_patch(rect_patch)

        arrow_color = '#FF6A00'
        if trajectory:
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-', color='#FFD700', linewidth=1, label='Trajectory')
            for i in range(len(trajectory)):
                if i == 0 or i == len(trajectory) - 1 or i % arrow_interval == 0:
                    x, y = trajectory[i, 0], trajectory[i, 1]
                    angle = car_angles[i] + np.pi / 2
                    dx = np.cos(angle) * 3
                    dy = np.sin(angle) * 5

                    arrow_start_x = x - dx * 0.3
                    arrow_start_y = y - dy * 0.3

                    plt.arrow(arrow_start_x, arrow_start_y, dx, dy,
                              head_width=3, head_length=4, fc=arrow_color, ec=arrow_color)
        # Legend
        grass_patch = patches.Patch(color=green_color, label='Off-Track Area (Grass)')
        track_patch = patches.Patch(color='#666666', label='Track')
        border_patch = patches.Patch(color='red', label='Curbing (red-white pattern at sharp turns)')
        view_patch = patches.Patch(color=view_color, alpha=0.1, label="Agent's Dynamic Visual Field")
        handles, labels = plt.gca().get_legend_handles_labels()
        custom_handles = [grass_patch, track_patch, border_patch, view_patch]
        all_handles = custom_handles + handles
        seen_labels = set()
        unique_handles = []
        for handle in all_handles:
            label = handle.get_label()
            if label not in seen_labels:
                seen_labels.add(label)
                unique_handles.append(handle)

        track_coverage = env.unwrapped.tile_visited_count / len(env.unwrapped.track) * 100

        plt.title(
            f"Track with Car Trajectory and Corresponding Dynamic View Areas\n"
            f"Track Completion Rate: {track_coverage:.1f} %")

        plt.axis('equal')
        plt.legend(handles=unique_handles)
        # =========================================================================


        # --- CONVERT PLOT TO BASE64 ---
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')  # "jpg", "svg" are all OK
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        # Cleanup
        plt.close()
        env.close()
        end_time = time.time()

        # Compile final info dictionary to pass back
        infos = {'done': done,
                 'truncated': truncated,
                 'episode_reward': episode_reward,
                 'track_coverage': track_coverage,
                 'episode_max_reward': episode_max_reward,
                'evaluate_time': end_time - start_time}

        # Must return the dictionary of metrics AND the base64 image string
        return infos, img_base64

    # =========================================================================
    # 🔒 BOILERPLATE - DO NOT MODIFY
    # Wrapper function for the evaluation engine.
    # =========================================================================
    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        ins_to_be_evaluated_id = kwargs.get('ins_to_be_evaluated_id', None)
        training_mode = kwargs.get('training_mode', True)
        return self.evaluate(callable_func, ins_to_be_evaluated_id, training_mode)
