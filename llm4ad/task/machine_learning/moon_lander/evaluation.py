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


# moon lander website  https://gymnasium.farama.org/environments/box2d/lunar_lander/
from __future__ import annotations

from typing import Optional, Tuple, List, Any, Set
import gymnasium as gym
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side plotting
import matplotlib.pyplot as plt
import io
import base64
import copy
import warnings
import time

from llm4ad.base import Evaluation
# =========================================================================
# 🛠️ USER DEFINED: Import your custom template and task description here
# =========================================================================
from llm4ad.task.machine_learning.moon_lander.template import template_program, task_description, \
    non_image_representation_explanation

__all__ = ['MoonLanderEvaluation']

class MoonLanderEvaluation(Evaluation):
    """Evaluator for the Lunar Lander control problem."""

    def __init__(self, whocall='Eoh', max_steps=200, timeout_seconds=300, **kwargs):
        """
            Args:
                - 'max_steps' (int): Maximum number of steps allowed per episode in the MountainCar-v0 environment (default is 500).
                - '**kwargs' (dict): Additional keyword arguments passed to the parent class initializer.

            Attributes:
                - 'env' (gym.Env): The MountainCar-v0 environment with a modified maximum episode length.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )
        self.whocall = whocall

        # =========================================================================
        # 🛠️ USER DEFINED (i): Environment Configuration
        # Modify these variables to configure your custom environment physics/rules.
        # =========================================================================
        self.env_name = 'LunarLander-v3'
        self.env_max_episode_steps = max_steps
        objective_value = kwargs.get('objective_value', 230)
        self.final_objective_score = objective_value
        self.non_image_representation_explanation = non_image_representation_explanation

        # Custom Lunar Lander specific configurations
        self.gravity = kwargs.get('gravity', -10.0)
        self.enable_wind = kwargs.get('enable_wind', False)
        self.wind_power = kwargs.get('wind_power', 15.0)
        self.turbulence_power = kwargs.get('turbulence_power', 1.5)

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

        # Initialize features (Custom step for DyCA Method)
        self.instance_feature = {}
        self.to_be_solve_ins_feature = {}
        self._generate_instance_features()  # If you have

    def evaluate(self, action_select: callable, ins_to_be_evaluated_id: Set | List | None = None, training_mode=True) -> \
            Optional[dict]:
        """
        🔒 MOSTLY BOILERPLATE: Aggregates results across instances.
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
        observations = {}
        num_episodes = len(ins_to_be_evaluated_id)
        episodes_recorder = {}

        total_fuel = 0
        success_count = 0

        # --- Serial Evaluation Loop ---

        for ins_id in ins_to_be_evaluated_id:
            env_seed = ins_to_be_evaluated_set[ins_id]
            each_evaluate_result = self.evaluate_single(action_select, env_seed)

            if each_evaluate_result is not None:
                infos, img_canvas = each_evaluate_result
                total_rewards[ins_id] = infos['episode_reward']
                total_fuel += infos['episode_fuel']
                image64s[ins_id] = img_canvas       # Canvas data to be converted later
                observations[ins_id] = infos['observations']

                if infos['episode_reward'] >= 200:
                    success_count += 1

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

        # --- Calculate Custom Aggregated Metrics ---
        mean_reward = np.mean(list(total_rewards.values()))
        mean_fuel = total_fuel / num_episodes
        success_rate = success_count / num_episodes
        min_reward_id = min(total_rewards, key=total_rewards.get)
        chosen_image = image64s[min_reward_id]
        observation_chosen = observations[min_reward_id]

        # Normalized Weighted Score (NWS)
        nws = (mean_reward / 200) * 0.6 + (1 - min(mean_fuel / 100, 1)) * 0.2 + success_rate * 0.2

        sorted_keys = sorted(instance_performance.keys())
        list_performance = [instance_performance[k]['score'] for k in sorted_keys]

        # =========================================================================
        # 🛠️ USER DEFINED (iii): Final Return Formatting
        # You MUST return 'score' and 'image' for MLES.
        # =========================================================================
        if self.whocall == 'mles':
            # Create base64 representation of the canvas here
            encoded_base64 = self.create_base64(chosen_image, nws, episodes_recorder, min_reward_id)
            observation_chosen_str = str(observation_chosen)
            test_result = {
                'Mean Reward': mean_reward,
                'Mean Fuel': mean_fuel,
                'Success Rate': success_rate,
                'NWS': nws
            }
            return {
                    # REQUIRED BY MLES:
                    'score': nws,
                    'image': encoded_base64,

                    # CUSTOM USER METRICS:
                    'observation': observation_chosen_str,
                    'Test result': test_result,
                    'all_ins_performance': instance_performance,
                    'list_performance': list_performance
                    }

        elif self.whocall == 'dyca':
            return {'all_ins_performance': instance_performance,
                    'list_performance': list_performance}  # {int ID:{'score': 0.1, 'evaluation_time':2}, ...}
        else:
            return nws

    def evaluate_single(self, action_select: callable, env_seed=42):
        """
        # =========================================================================
        # 🛠️ USER DEFINED (ii): Single Episode Evaluation & Image Generation
        # Run the environment, track fuel/rewards, and generate the image canvas.
        # =========================================================================
        """
        start_time = time.time()
        env = gym.make(self.env_name, render_mode='rgb_array',
                       gravity=self.gravity,
                       enable_wind=self.enable_wind,
                       wind_power=self.wind_power,
                       turbulence_power=self.turbulence_power)
        observation, _ = env.reset(seed=env_seed)  # gym initialization
        action = 0  # initial action
        episode_reward = 0
        episode_fuel = 0

        # Create a blank canvas to overlay trajectory frames
        canvas = np.zeros((400, 600, 3), dtype=np.float32)
        observations = []

        pre_observation = copy.deepcopy(observation)
        observation, reward, done, truncated, info = env.step(action)

        flash_calculator = 0
        for i in range(self.env_max_episode_steps + 1):  # protect upper limits
            action = action_select(observation,
                                   action,
                                   pre_observation)
            pre_observation = copy.deepcopy(observation)
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # Track fuel usage based on engine firing actions
            if action in [1, 2, 3]:
                episode_fuel += 1

            # Render frame and create transparent overlay for trajectory history
            if flash_calculator >= 10:
                img = env.render()
                mask = np.any(img != [0, 0, 0], axis=-1)
                alpha = min(i / self.env_max_episode_steps, 1.0)

                canvas[mask] = canvas[mask] * (1 - alpha) + img[mask] * alpha

                observation_str = ', '.join([f"{x:.3f}" for x in observation])
                observations.append(f"[{observation_str}]")
                flash_calculator = 0

            flash_calculator += 1

            if done or truncated or i == self.env_max_episode_steps:
                img = env.render()
                mask = np.any(img != [0, 0, 0], axis=-1)
                alpha = i / self.env_max_episode_steps  # 假设最大步数为200，可以根据实际情况调整
                alpha = min(alpha, 1.0)  # 确保透明度不超过1
                canvas[mask] = canvas[mask] * (1 - alpha) + img[mask] * alpha
                observation_str = ', '.join([f"{x:.3f}" for x in observation])
                observations.append(f"[{observation_str}]")
                # fitness = abs(observation[0]) + abs(yv[-2]) - (observation[6] + observation[7])
                env.close()
                end_time = time.time()
                infos = {'done': done,
                         'truncated': truncated,
                         'episode_fuel': episode_fuel,
                         'episode_reward': episode_reward,
                         'observations': observations,
                         'evaluate_time': end_time - start_time}

                # Return the custom metrics and the raw numpy canvas
                return infos, canvas

    # =========================================================================
    # 🔒 BOILERPLATE - DO NOT MODIFY
    # Wrapper function for the evaluation engine.
    # =========================================================================
    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        ins_to_be_evaluated_id = kwargs.get('ins_to_be_evaluated_id', None)
        training_mode = kwargs.get('training_mode', True)
        return self.evaluate(callable_func, ins_to_be_evaluated_id, training_mode)

    # =========================================================================
    # 🛠️ USER DEFINED (iv): Custom Task-Specific Methods
    # Add any extra helper functions, feature extractors, or visualizers here.
    # =========================================================================
    def create_base64(self, original_input, fitness, recoder, which_image):
        """Helper to convert the numpy canvas into a base64 encoded string for MLES."""
        img_bytes = io.BytesIO()
        plt.imshow(original_input.astype(np.uint8))
        image_recode = recoder[which_image]

        if image_recode['done']:
            final_state = "Landed safely"
        elif image_recode['truncated']:
            final_state = "Crashed"
        else:
            final_state = "Landing failed"

        plt.title(f'Lander Trajectory over 200 steps\n Score: {fitness:.3f} | Final State: {final_state}')
        plt.axis('off')

        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        # 对图像进行base64编码
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        return img_base64

    # =========================================================================
    # 🛑 OPTIONAL / ADVANCED: Custom Analytics & Visualization (SAFE TO IGNORE)
    # -------------------------------------------------------------------------
    # NOTE FOR USERS: Everything below this line is highly specific to the
    # analysis and trajectory clustering we performed for the MLES paper.
    #
    # You DO NOT need to read, understand, or implement anything like this
    # to get your own custom environment working!
    # =========================================================================
    def feature_pipeline(self, seed, env_max_episode_steps=100):
        """(Advanced) No action → feature extraction for clustering."""
        env = gym.make('LunarLander-v3', render_mode='rgb_array',
                       gravity=-10,
                       enable_wind=False,
                       wind_power=15,
                       turbulence_power=1.5)
        observation, _ = env.reset(seed=seed)  # initialization
        action = 0
        observations = []
        flash_calculator = 0
        for i in range(env_max_episode_steps + 1):  # protect upper limits
            observation, reward, done, truncated, info = env.step(action)
            if flash_calculator >= 5:
                observations.append(observation)
                flash_calculator = 0
            flash_calculator += 1
        observations = np.array(observations)
        env.close()
        feature_x = observations[:, 0]
        feature_y = observations[:, 1]
        feature = np.concatenate((feature_x, feature_y)) * 10
        return feature.tolist()

    def _generate_instance_features(self):
        """Generate features for all instances."""
        if self.instance_feature:
            warnings.warn("Training instance features already exist.")
        if self.to_be_solve_ins_feature:
            warnings.warn("Testing instance features already exist.")

        self.instance_feature.clear()
        self.to_be_solve_ins_feature.clear()
        for instance_id, config in self.instance_set.items():
            feature = self.feature_pipeline(config)
            self.instance_feature[instance_id] = feature
        for instance_id, config in self.ins_to_be_solve_set.items():
            feature = self.feature_pipeline(config)
            self.to_be_solve_ins_feature[instance_id] = feature

    def visualize_instance_features_base64(self, mode: str = 'combined') -> str:
        """(Advanced) Visualization tool for trajectory features."""
        from matplotlib.lines import Line2D

        valid_modes = ['combined', 'training', 'testing']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

        fig, ax = plt.subplots(figsize=(12, 9))

        plot_any = False
        legend_elements = []

        if mode in ['combined', 'training']:
            plot_training = False
            for instance_id, feature in self.instance_feature.items():
                if not feature or len(feature) < 2:
                    continue
                plot_any = True
                plot_training = True

                L = len(feature)
                mid_point = L // 2
                x_coords = feature[0: mid_point]
                y_coords = feature[mid_point: L]

                if not x_coords or not y_coords:
                    continue

                ax.plot(x_coords, y_coords, color='blue', alpha=0.4, linestyle=':')
                ax.plot(x_coords[-1], y_coords[-1], 'o', color='blue', markersize=4)
                ax.text(x_coords[-1], y_coords[-1] + 0.05, f'{instance_id}', color='blue',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

            if plot_training:
                legend_elements.append(
                    Line2D([0], [0], color='blue', lw=2, linestyle=':', label='Training Instance Trajectory'))
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', label='Training Landing Point (ID)', markerfacecolor='blue',
                           markersize=8))

        if mode in ['combined', 'testing']:
            plot_testing = False
            for instance_id, feature in self.to_be_solve_ins_feature.items():
                if not feature or len(feature) < 2:
                    continue
                plot_any = True
                plot_testing = True

                L = len(feature)
                mid_point = L // 2
                x_coords = feature[0: mid_point]
                y_coords = feature[mid_point: L]

                if not x_coords or not y_coords:
                    continue

                ax.plot(x_coords, y_coords, color='orange', alpha=0.4, linestyle=':')
                ax.plot(x_coords[-1], y_coords[-1], 'o', color='orange', markersize=4)
                ax.text(x_coords[-1], y_coords[-1] + 0.05, f'{instance_id}', color='darkorange',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

            if plot_testing:
                legend_elements.append(
                    Line2D([0], [0], color='orange', lw=2, linestyle=':', label='Testing Instance Trajectory'))
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Testing Landing Point (ID)',
                                              markerfacecolor='orange', markersize=8))

        title_suffix = {
            'combined': 'Training & Testing Instances',
            'training': 'Training Instances Only',
            'testing': 'Testing Instances Only'
        }
        ax.set_title(f'Instance Features: No-Action Trajectories\n({title_suffix[mode]})', fontsize=16)
        ax.set_xlabel('X Coordinate (scaled * 10)', fontsize=12)
        ax.set_ylabel('Y Coordinate (scaled * 10)', fontsize=12)

        ax.axhline(0, color='grey', linestyle='--', linewidth=2)
        ax.plot([-2, 2], [0, 0], color='red', linewidth=4)
        legend_elements.append(Line2D([0], [0], color='red', lw=4, label='Landing Pad (y=0, x=[-2, 2])'))

        ax.set_xlim(-10, 10)
        ax.set_ylim(bottom=-1)
        ax.grid(True, linestyle=':', alpha=0.6)

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')

        if not plot_any:
            ax.text(0.5, 0.5, f"No instance features found for mode '{mode}'.",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')

        plt.tight_layout()

        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight')
        plt.close(fig)
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

        return img_base64

    def show_instance_features(self, mode: str = 'combined', duichen = False):
        """(Advanced) Interactive UI for instance features."""
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("'TkAgg' backend not found, trying default interactive backend...")
            matplotlib.use(matplotlib.get_backend())
            import matplotlib.pyplot as plt

        from matplotlib.lines import Line2D

        valid_modes = ['combined', 'training', 'testing']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

        fig, ax = plt.subplots(figsize=(12, 9))

        plot_any = False
        legend_elements = []

        if mode in ['combined', 'training']:
            plot_training = False
            for instance_id, feature in self.instance_feature.items():
                if not feature or len(feature) < 2:
                    continue
                plot_any = True
                plot_training = True

                L = len(feature)
                mid_point = L // 2
                x_coords = feature[0: mid_point]
                y_coords = feature[mid_point: L]

                if not x_coords or not y_coords:
                    continue

                ax.plot(x_coords, y_coords, color='blue', alpha=0.4, linestyle=':')
                ax.plot(x_coords[-1], y_coords[-1], 'o', color='blue', markersize=4)
                ax.text(x_coords[-1], y_coords[-1] + 0.05, f'{instance_id}', color='blue',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

                if duichen:
                    fu_x = [-xc for xc in x_coords]
                    ax.plot(fu_x, y_coords, color='blue', alpha=0.4, linestyle=':')
                    ax.plot(fu_x[-1], y_coords[-1], 'o', color='blue', markersize=4)
                    ax.text(fu_x[-1], y_coords[-1] + 0.05, f'{instance_id}', color='blue',
                            ha='center', va='bottom', fontsize=7, fontweight='bold')

            if plot_training:
                legend_elements.append(
                    Line2D([0], [0], color='blue', lw=2, linestyle=':', label='Training Instance Trajectory'))
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', label='Training Landing Point (ID)', markerfacecolor='blue',
                           markersize=8))

        if mode in ['combined', 'testing']:
            plot_testing = False
            for instance_id, feature in self.to_be_solve_ins_feature.items():
                if not feature or len(feature) < 2:
                    continue
                plot_any = True
                plot_testing = True

                L = len(feature)
                mid_point = L // 2
                x_coords = feature[0: mid_point]
                y_coords = feature[mid_point: L]

                if not x_coords or not y_coords:
                    continue

                ax.plot(x_coords, y_coords, color='orange', alpha=0.4, linestyle=':')
                ax.plot(x_coords[-1], y_coords[-1], 'o', color='orange', markersize=4)
                ax.text(x_coords[-1], y_coords[-1] + 0.05, f'{instance_id}', color='darkorange',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

            if plot_testing:
                legend_elements.append(
                    Line2D([0], [0], color='orange', lw=2, linestyle=':', label='Testing Instance Trajectory'))
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Testing Landing Point (ID)',
                                              markerfacecolor='orange', markersize=8))

        title_suffix = {
            'combined': 'Training & Testing Instances',
            'training': 'Training Instances Only',
            'testing': 'Testing Instances Only'
        }
        ax.set_title(f'Instance Features: No-Action Trajectories\n({title_suffix[mode]})', fontsize=16)
        ax.set_xlabel('X Coordinate (scaled * 10)', fontsize=12)
        ax.set_ylabel('Y Coordinate (scaled * 10)', fontsize=12)

        ax.axhline(0, color='grey', linestyle='--', linewidth=2)
        ax.plot([-2, 2], [0, 0], color='red', linewidth=4)
        legend_elements.append(Line2D([0], [0], color='red', lw=4, label='Landing Pad (y=0, x=[-2, 2])'))

        ax.set_xlim(-10, 10)
        ax.set_ylim(bottom=-1)
        ax.grid(True, linestyle=':', alpha=0.6)

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')

        if not plot_any:
            ax.text(0.5, 0.5, f"No instance features found for mode '{mode}'.",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')

        plt.tight_layout()

        print("Displaying plot window... (Close the window to continue the script)")
        plt.show()

        plt.close(fig)

    def show_clustered_features(self, json_file_path: str, data_source: str, cluster_key: str):
        """
        (Advanced) Visualizes clustered behavioral features based on JSON data.

        根据 JSON 文件中的聚类/来源分配，可视化实例特征并为每个聚类/来源指定不同颜色。
        这是一个高度灵活的函数，具有两个独立的控制开关：
        1. 画图用的数据 (data_source)
        2. 读的JSON参数 (cluster_key)

        参数:
            json_file_path (str):
                JSON 文件的路径。
                (例如: './log_dir/final_output.json' 或 './log_dir/using_final_results.json')

            data_source (str):
                指定要绘制哪组实例的特征 (数据源)。
                - 'training': 使用 self.instance_feature
                - 'testing':  使用 self.to_be_solve_ins_feature

            cluster_key (str):
                指定从 JSON 中读取哪个键来分配颜色 (聚类/来源)。
                - 'cluster_id_instances' (通常来自训练JSON)
                - 'apply_cluster_of_each_instance' (通常来自使用JSON)
                - 'match_cluster_of_each_instance' (通常来自使用JSON)
        """
        import json
        import warnings
        from matplotlib.lines import Line2D
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("'TkAgg' backend not found, trying default interactive backend...")
            matplotlib.use(matplotlib.get_backend())
            import matplotlib.pyplot as plt

        if data_source == 'training':
            feature_dict = self.instance_feature
        elif data_source == 'testing':
            feature_dict = self.to_be_solve_ins_feature
        else:
            raise ValueError(f"Invalid data_source '{data_source}'. Must be 'training' or 'testing'.")

        if not feature_dict:
            warnings.warn(f"No features found for data_source '{data_source}'. Nothing to plot.")
            return

        cluster_assignments_raw = None

        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)

            valid_keys = ['cluster_id_instances', 'apply_cluster_of_each_instance', 'match_cluster_of_each_instance']
            if cluster_key not in valid_keys:
                raise ValueError(f"Invalid cluster_key '{cluster_key}'. Must be one of {valid_keys}")

            cluster_assignments_raw = data.get(cluster_key)

            if not cluster_assignments_raw:
                print(f"Error: Key '{cluster_key}' not found or is empty in {json_file_path}.")
                return

            cluster_assignments = {}
            for key, id_list in cluster_assignments_raw.items():
                try:
                    cluster_assignments[str(key)] = [int(instance_id) for instance_id in id_list]
                except (ValueError, TypeError) as e:
                    warnings.warn(f"Skipping cluster '{key}'. Could not convert instance IDs to int: {e}")

            if not cluster_assignments:
                print("Error: No valid cluster assignments found after processing.")
                return

        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_file_path}")
            return
        except Exception as e:
            print(f"Error reading or parsing JSON file: {e}")
            traceback.print_exc()
            return

        preset_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
            '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d',
            '#9edae5', '#a55194', '#393b79'
        ]
        color_map = {}
        legend_elements = []

        DEFAULT_OTHER_COLOR = '#7f7f7f'

        all_cluster_keys = sorted(cluster_assignments.keys())

        for cluster_key in all_cluster_keys:
            color = ''
            label = f'Cluster: {cluster_key}'

            if cluster_key == 'main':
                color = 'black'
            elif cluster_key.isdigit():
                cluster_num = int(cluster_key)
                index = cluster_num % len(preset_colors)
                color = preset_colors[index]
            else:
                color = DEFAULT_OTHER_COLOR

            if color not in color_map.values() or color == 'black':
                legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))
            elif color == DEFAULT_OTHER_COLOR and not any(
                    l.get_label() == 'Cluster: Other (grey)' for l in legend_elements):
                legend_elements.append(Line2D([0], [0], color=color, lw=2, label='Cluster: Other (grey)'))

            color_map[cluster_key] = color

        legend_elements.sort(key=lambda x: x.get_label())
        instance_to_color_map = {}
        for cluster_key, instance_list in cluster_assignments.items():
            color = color_map[cluster_key]
            for instance_id in instance_list:
                instance_to_color_map[instance_id] = color

        fig, ax = plt.subplots(figsize=(12, 9))
        plot_any = False

        for instance_id, feature in feature_dict.items():
            plot_color = instance_to_color_map.get(instance_id)
            if plot_color is None: continue
            if not feature or len(feature) < 2: continue
            plot_any = True

            L = len(feature);
            mid_point = L // 2
            x_coords = feature[0: mid_point];
            y_coords = feature[mid_point: L]
            if not x_coords or not y_coords: continue

            ax.plot(x_coords, y_coords, color=plot_color, alpha=0.6, linestyle='-')

            ax.text(x_coords[-1], y_coords[-1] + 0.05, f'{instance_id}', color=plot_color,
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_title(
            f'Clustered Instance Features\n'
            f'Data Source: "{data_source}" | Cluster Key: "{cluster_key}"\n'
            f'File: {json_file_path}',
            fontsize=14
        )
        ax.set_xlabel('X Coordinate (scaled * 10)', fontsize=12)
        ax.set_ylabel('Y Coordinate (scaled * 10)', fontsize=12)

        ax.axhline(0, color='grey', linestyle='--', linewidth=2)
        ax.plot([-2, 2], [0, 0], color='red', linewidth=4)
        legend_elements.append(Line2D([0], [0], color='red', lw=4, label='Landing Pad'))

        ax.set_xlim(-10, 10);
        ax.set_ylim(bottom=-1)
        ax.grid(True, linestyle=':', alpha=0.6)

        if legend_elements:
            # ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0))
            ax.legend(handles=legend_elements, loc='best')

        if not plot_any:
            ax.text(0.5, 0.5,
                    f"No instances from '{data_source}' feature set\nwere found in the JSON cluster map (Key: {cluster_key}).",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')

        # plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.tight_layout()

        print(
            f"Displaying clustered plot window...\n  Data Source: '{data_source}'\n  Cluster Key: '{cluster_key}'\n(Close the window to continue the script)")
        plt.show()

        plt.close(fig)

    def filter_custom_instances_by_x(self, instance_input: list | dict, threshold: float = 4.5) -> tuple[list, list]:
        """
        (Advanced) Custom filter logic.
        输入任意列表或字典，现场生成特征，并将实例分为两拨：
        1. 满足条件（X 越界）
        2. 不满足条件（X 正常）

        参数:
            instance_input: List[int] (Seeds) 或 Dict{id: seed}
            threshold (float): X 坐标的阈值，默认 4.5。

        返回:
            tuple: (satisfied_ids, not_satisfied_ids)
                   - satisfied_ids: X < -threshold 或 X > threshold 的实例
                   - not_satisfied_ids: X 在 [-threshold, threshold] 之间的实例
        """
        target_dict = {}

        if isinstance(instance_input, list):
            target_dict = {i: i for i in instance_input}
        elif isinstance(instance_input, dict):
            target_dict = instance_input
        else:
            raise ValueError("Input must be a list of seeds or a dict of {id: seed}.")

        print(f"🚀 Start filtering {len(target_dict)} custom instances (Threshold: +/-{threshold})...")

        satisfied_ids = []
        not_satisfied_ids = []

        count = 0
        for inst_id, seed in target_dict.items():
            count += 1
            print(f"   -> Processing {count}/{len(target_dict)}...", end='\r')

            try:
                feature = self.feature_pipeline(seed=seed)

                if not feature:
                    print(f"   -> ⚠️ Failed to generate feature for seed {seed}")
                    continue

                mid = len(feature) // 2
                x_coords = feature[:mid]

                is_extreme = False
                for x in x_coords:
                    if x < -threshold or x > threshold:
                        is_extreme = True
                        break

                if is_extreme:
                    satisfied_ids.append(inst_id)
                else:
                    not_satisfied_ids.append(inst_id)

            except Exception as e:
                print(f"   -> ❌ Error processing seed {seed}: {e}")

        print(f"\n✅ Done.")
        print(f"   -> Satisfied (Extreme X): {len(satisfied_ids)}")
        print(f"   -> Not Satisfied (Normal X): {len(not_satisfied_ids)}")

        return satisfied_ids, not_satisfied_ids


if __name__ == '__main__':

    seeds = [6, 9, 17, 29, 57,
             44, 18, 69, 26, 68,
             65, 23, 51, 93, 16,
             87, 92, 90, 22, 73,
             60, 10, 19, 97, 11,
             14, 99, 98, 8, 28,
             43, 56, 89, 15, 74]

    # seeds_order = [6, 9, 29,57, 44,
    #                17, 69, 18, ]
    # Training
    # seeds = [i for i in range(20)]
    instance_set = {}
    for id, seed in enumerate(seeds):
        instance_set[id] = seed
    algo_seed_path = './init_pop_size16.json'

    # Using
    using_algo_designed_path = ""
    Using_seeds = [i for i in range(100, 150)]
    # Using_seeds = seeds
    ins_to_be_solve_set = {}
    for id, seed in enumerate(Using_seeds):
        ins_to_be_solve_set[id] = seed

    run_mode = 'Combined'
    task = MoonLanderEvaluation(whocall='dyca', instance_set=instance_set, run_mode=run_mode,
                                ins_to_be_solve_set=ins_to_be_solve_set)

    # --- show_clustered_features ---
    import os

    print("\nGenerating CLUSTERED feature visualization...")

    # !!! 警告: 你必须修改这些路径 !!!
    # 路径1: 训练JSON (包含 'cluster_id_instances')
    training_json_path = r'C:\0_QL_work\015_DyEvo\DyEvo\example\moon_lander\logs\20251126_223929\designed_result\final_output.json'  # <--- 修改这里

    # 路径2: 使用JSON (包含 'apply_cluster_of_each_instance')
    using_json_path_training = r'C:\0_QL_work\015_DyEvo\DyEvo\example\moon_lander\logs\20251126_223929\using\20251127_144549_U\using_final_output.json'
    using_json_path_testing = r'C:\0_QL_work\015_DyEvo\DyEvo\example\moon_lander\logs\20251126_223929\using\20251127_144923_U\using_final_output.json'  # <--- 修改这里

    # --- 调用 show_instance_features ---
    print("Generating instance feature visualization...")

    try:
        plot_mode = 'training'

        '''
        - 'combined': (默认) 绘制训练实例和待解实例。
        - 'training': 只绘制训练实例 (self.instance_feature)。
        - 'testing':  只绘制待解实例 (self.to_be_solve_ins_feature)。
        '''

        print(f"Generating plot for mode: '{plot_mode}'...")

        # 调用这个更新后的 show 函数
        task.show_instance_features(mode=plot_mode,duichen=True)

        print("Plot window closed.")

    except Exception as e:
        print(f"Failed to generate feature visualization: {e}")
        traceback.print_exc()

    # --- 示例 1: 查看 "training" 数据的 "训练聚类" 结果 ---
    try:
        if os.path.exists(training_json_path):
            task.show_clustered_features(
                json_file_path=training_json_path,
                data_source='training',  # <--- 数据
                cluster_key='cluster_id_instances'  # 训练数据被分类的情况
            )
            print("Training clustered plot window closed.")
        else:
            print(f"Warning: Training JSON file not found at {training_json_path}. Skipping plot 1.")
    except Exception as e:
        print(f"Failed to generate plot 1: {e}")

    try:        # 在训练集上main的作用如何？
        if os.path.exists(training_json_path):
            task.show_clustered_features(
                json_file_path=using_json_path_training,
                data_source='training',  # <--- 数据
                cluster_key='apply_cluster_of_each_instance'  # 训练数据应用的时候最终用的cluster
            )
            print("Training clustered plot window closed.")
        else:
            print(f"Warning: Training JSON file not found at {training_json_path}. Skipping plot 1.")
    except Exception as e:
        print(f"Failed to generate plot 1: {e}")

    # --- 示例 2: 查看 "testing" 数据的 "实际应用" 结果 (最常用) ---
    try:
        if os.path.exists(using_json_path_testing):
            task.show_clustered_features(
                json_file_path=using_json_path_testing,
                data_source='testing',  # <--- 数据
                cluster_key='apply_cluster_of_each_instance'  # 得到结果的最终用的cluster
            )
            print("Testing 'apply' plot window closed.")
        else:
            print(f"Warning: Using JSON file not found at {using_json_path_testing}. Skipping plot 2.")
    except Exception as e:
        print(f"Failed to generate plot 2: {e}")

    # --- 示例 3: 查看 "testing" 数据的 "KNN预测" 结果 (用于分析) ---
    try:
        if os.path.exists(using_json_path_testing):
            task.show_clustered_features(
                json_file_path=using_json_path_testing,
                data_source='testing',  # <--- 数据
                cluster_key='match_cluster_of_each_instance'  # 匹配到的cluster
            )
            print("Testing 'match' plot window closed.")
        else:
            print(f"Warning: Using JSON file not found at {using_json_path_testing}. Skipping plot 3.")
    except Exception as e:
        print(f"Failed to generate plot 3: {e}")

    print("Initialization complete.")
    print('aaa')
