# Module Name: MoonLanderEvaluation
# Last Revision: 2025/3/5
# Description: Implements a heuristic strategy function to guide a lunar lander to achieve safe landings
#              at the center of the target area. The function selects actions based on the lander's
#              current state, aiming to minimize the number of steps required for a safe landing.
#              A "safe landing" is defined as a touchdown with minimal vertical velocity, upright
#              orientation, and angular velocity and angle close to zero.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    -   x_coordinate: float - x coordinate, range [-1, 1] (default: None).
#    -   y_coordinate: float - y coordinate, range [-1, 1] (default: None).
#    -   x_velocity: float - x velocity (default: None).
#    -   x_velocity: float - y velocity (default: None).
#    -   angle: float - angle (default: None).
#    -   angular_velocity: float - angular velocity (default: None).
#    -   l_contact: int - 1 if the first leg has contact, else 0 (default: None).
#    -   r_contact: int - 1 if the second leg has contact, else 0 (default: None).
#    -   last_action: int - last action taken by the lander, values [0, 1, 2, 3] (default: None).
#    -   timeout_seconds: int - Maximum allowed time (in seconds) for the evaluation process (default: 20).
#
# References:
#   - Brockman, Greg, et al. "Openai gym." arXiv preprint arXiv:1606.01540 (2016).
#
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

from llm4ad.base import Evaluation
from llm4ad.task.machine_learning.moon_lander.template import template_program, task_description, \
    non_image_representation_explanation

import traceback
import matplotlib

matplotlib.use('Agg')  # 选择不显示的后端
import matplotlib.pyplot as plt
import io
from io import BytesIO
import base64
import copy

import warnings
import time

__all__ = ['MoonLanderEvaluation']


# def evaluate(env: gym.Env, action_select: callable) -> float | None:


class MoonLanderEvaluation(Evaluation):
    """Evaluator for moon lander problem."""

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

        self.env_name = 'LunarLander-v3'
        self.env_max_episode_steps = max_steps
        self.whocall = whocall
        objective_value = kwargs.get('objective_value', 230)
        self.final_objective_score = objective_value
        self.non_image_representation_explanation = non_image_representation_explanation

        # --- 实例集处理 ---
        self._mode = kwargs.get('run_mode', 'Training')
        self.instance_set = kwargs.get('instance_set')
        self.instance_id_set = tuple(self.instance_set.keys())
        if self._mode == 'Training' and not self.instance_set:
            # 使用更标准的 Python 异常处理
            raise ValueError("没有提供Training实例集 (instance_set)，无法进行评估。")

        self.ins_to_be_solve_set = kwargs.get('ins_to_be_solve_set')
        self.to_be_solve_instance_id_set = tuple(self.ins_to_be_solve_set.keys())
        if self._mode == 'Using' and not self.ins_to_be_solve_set:
            # 使用更标准的 Python 异常处理
            raise ValueError("没有提供Testing实例集 (ins_to_be_solve_set)，无法进行评估。")

        if self._mode == 'Combined' and (not self.instance_set or not self.ins_to_be_solve_set):
            # 使用更标准的 Python 异常处理
            raise ValueError("缺少Training或Testing实例集 ，无法进行评估。")

        self.gravity = kwargs.get('gravity', -10.0)
        self.enable_wind = kwargs.get('enable_wind', False)
        self.wind_power = kwargs.get('wind_power', 15.0)
        self.turbulence_power = kwargs.get('turbulence_power', 1.5)

        self.instance_feature = {}
        self.to_be_solve_ins_feature = {}
        self._generate_instance_features()  # If you have

    def feature_pipeline(self, seed, env_max_episode_steps=100):
        """No action → feature."""
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
        """为所有实例生成特征，并存储在 instance_feature 属性中。"""
        if self.instance_feature:
            warnings.warn("训练实例特征已存在，将重新生成。")
        if self.to_be_solve_ins_feature:
            warnings.warn("待求解实例特征已存在，将重新生成。")

        self.instance_feature.clear()
        self.to_be_solve_ins_feature.clear()
        for instance_id, config in self.instance_set.items():
            feature = self.feature_pipeline(config)
            self.instance_feature[instance_id] = feature
        for instance_id, config in self.ins_to_be_solve_set.items():
            feature = self.feature_pipeline(config)
            self.to_be_solve_ins_feature[instance_id] = feature

    def create_base64(self, original_input, fitness, recoder, which_image):
        """
        这个函数将图片变成base64的形式，最终输出base64
        """
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

    def evaluate(self, action_select: callable, ins_to_be_evaluated_id: Set | List | None = None, training_mode=True) -> \
            Optional[dict]:
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

        # --- L3 串行评估 (Serial Evaluation) ---
        # print(f"MoonLander: Starting SERIAL evaluation of {num_episodes} instances...")

        for ins_id in ins_to_be_evaluated_id:
            env_seed = ins_to_be_evaluated_set[ins_id]

            # 直接调用 evaluate_single，不再提交到线程池
            each_evaluate_result = self.evaluate_single(action_select, env_seed)

            if each_evaluate_result is not None:
                infos = each_evaluate_result[0]

                # --- 数据记录 (串行模式下不需要锁) ---
                total_rewards[ins_id] = infos['episode_reward']
                total_fuel += infos['episode_fuel']
                image64s[ins_id] = each_evaluate_result[1]
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

        # --- 聚合 (保持不变) ---
        # 检查是否有任何有效结果
        if not total_rewards:
            print("Evaluation failed: No valid rewards were collected.")
            return None

        mean_reward = np.mean(list(total_rewards.values()))
        mean_fuel = total_fuel / num_episodes
        success_rate = success_count / num_episodes

        min_reward_id = min(total_rewards, key=total_rewards.get)
        chosen_image = image64s[min_reward_id]
        observation_chosen = observations[min_reward_id]

        # 标准化加权得分（权重α=0.6, β=0.2, γ=0.2）
        nws = (mean_reward / 200) * 0.6 + (1 - min(mean_fuel / 100, 1)) * 0.2 + success_rate * 0.2

        sorted_keys = sorted(instance_performance.keys())
        # 返回按 key 排序的 score 列表
        list_performance = [instance_performance[k]['score'] for k in sorted_keys]

        if self.whocall == 'mles':
            encoded_base64 = self.create_base64(chosen_image, nws, episodes_recorder, min_reward_id)
            observation_chosen_str = str(observation_chosen)

            test_result = {
                'Mean Reward': mean_reward,
                'Mean Fuel': mean_fuel,
                'Success Rate': success_rate,
                'NWS': nws
            }
            return {'score': nws,
                    'image': encoded_base64,
                    'observation': observation_chosen_str,

                    'Test result': test_result,
                    'all_ins_performance': instance_performance,
                    'list_performance': list_performance
                    }

        # elif self.whocall in ['eoh', 'reevo', 'funsearch']:
        #     return (mean_reward, instance_performance)
        elif self.whocall == 'dyca':
            return {'all_ins_performance': instance_performance,
                    'list_performance': list_performance}  # {int ID:{'score': 0.1, 'evaluation_time':2}, ...}
        else:
            return nws

    def evaluate_single(self, action_select: callable, env_seed=42):
        """Evaluate heuristic function on moon lander problem."""
        start_time = time.time()
        env = gym.make(self.env_name, render_mode='rgb_array',
                       gravity=self.gravity,
                       enable_wind=self.enable_wind,
                       wind_power=self.wind_power,
                       turbulence_power=self.turbulence_power)
        observation, _ = env.reset(seed=env_seed)  # initialization
        action = 0  # initial action
        episode_reward = 0
        episode_fuel = 0

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
            if action in [1, 2, 3]:
                episode_fuel += 1

            if flash_calculator >= 10:
                img = env.render()
                # 提取非黑色部分
                mask = np.any(img != [0, 0, 0], axis=-1)
                # 计算动态透明度

                alpha = i / self.env_max_episode_steps  # 假设最大步数为200，可以根据实际情况调整
                alpha = min(alpha, 1.0)  # 确保透明度不超过1
                # 将当前帧的非黑色部分叠加到画布上
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
                return infos, canvas

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        ins_to_be_evaluated_id = kwargs.get('ins_to_be_evaluated_id', None)
        training_mode = kwargs.get('training_mode', True)
        return self.evaluate(callable_func, ins_to_be_evaluated_id, training_mode)

    def visualize_instance_features_base64(self, mode: str = 'combined') -> str:
        """
        生成实例特征 (无动作轨迹) 的可视化图像，并返回 Base64 编码的 PNG 字符串。

        参数:
            mode (str):
                - 'combined': (默认) 绘制训练实例和待解实例。
                - 'training': 只绘制训练实例 (self.instance_feature)。
                - 'testing':  只绘制待解实例 (self.to_be_solve_ins_feature)。

        返回:
            str: 包含 PNG 图像的 Base64 编码字符串。
        """
        # 需要额外导入 Line2D 来创建自定义图例
        from matplotlib.lines import Line2D

        # 1. 检查 mode 参数
        valid_modes = ['combined', 'training', 'testing']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

        # 2. 初始化 Matplotlib 画布
        fig, ax = plt.subplots(figsize=(12, 9))

        plot_any = False  # 标记是否绘制了任何内容
        legend_elements = []  # 用于动态构建图例

        # 3. 绘制训练实例 (Training Instances)
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

        # 4. 绘制待解实例 (Testing Instances)
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

        # 5. 设置图像美化和信息
        title_suffix = {
            'combined': 'Training & Testing Instances',
            'training': 'Training Instances Only',
            'testing': 'Testing Instances Only'
        }
        ax.set_title(f'Instance Features: No-Action Trajectories\n({title_suffix[mode]})', fontsize=16)
        ax.set_xlabel('X Coordinate (scaled * 10)', fontsize=12)
        ax.set_ylabel('Y Coordinate (scaled * 10)', fontsize=12)

        # 着陆坪
        ax.axhline(0, color='grey', linestyle='--', linewidth=2)
        ax.plot([-2, 2], [0, 0], color='red', linewidth=4)
        legend_elements.append(Line2D([0], [0], color='red', lw=4, label='Landing Pad (y=0, x=[-2, 2])'))

        ax.set_xlim(-10, 10)
        ax.set_ylim(bottom=-1)
        ax.grid(True, linestyle=':', alpha=0.6)

        # 6. 添加图例
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')

        if not plot_any:
            ax.text(0.5, 0.5, f"No instance features found for mode '{mode}'.",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')

        plt.tight_layout()

        # 7. 将图像保存到内存并编码为 Base64
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight')
        plt.close(fig)  # 关闭图像
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

        return img_base64

    def show_instance_features(self, mode: str = 'combined', duichen = False):
        """
        生成实例特征 (无动作轨迹) 的可视化图像，并直接在窗口中显示。

        参数:
            mode (str):
                - 'combined': (默认) 绘制训练实例和待解实例。
                - 'training': 只绘制训练实例 (self.instance_feature)。
                - 'testing':  只绘制待解实例 (self.to_be_solve_ins_feature)。
        """

        # --- 新增：强制切换到交互式后端 ---
        try:
            # 必须在导入 pyplot 之前设置后端
            import matplotlib
            matplotlib.use('TkAgg')  # 尝试使用 'TkAgg' (需要 Tkinter)
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("'TkAgg' backend not found, trying default interactive backend...")
            # 恢复到 Matplotlib 的默认后端（希望能是交互式的）
            matplotlib.use(matplotlib.get_backend())
            import matplotlib.pyplot as plt
        # --- 结束新增 ---

        from matplotlib.lines import Line2D

        # 1. 检查 mode 参数
        valid_modes = ['combined', 'training', 'testing']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")

        # 2. 初始化 Matplotlib 画布
        fig, ax = plt.subplots(figsize=(12, 9))

        plot_any = False  # 标记是否绘制了任何内容
        legend_elements = []  # 用于动态构建图例

        # 3. 绘制训练实例 (Training Instances)
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

        # 4. 绘制待解实例 (Testing Instances)
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

        # 5. 设置图像美化和信息
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

        # 6. 添加图例
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')

        if not plot_any:
            ax.text(0.5, 0.5, f"No instance features found for mode '{mode}'.",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')

        plt.tight_layout()

        # 7. 显示图像
        print("Displaying plot window... (Close the window to continue the script)")
        plt.show()

        # 显示后关闭图形，释放内存
        plt.close(fig)

    def show_clustered_features(self, json_file_path: str, data_source: str, cluster_key: str):
        """
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

        # --- 1. 导入所需库 ---
        import json
        import warnings
        from matplotlib.lines import Line2D
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # 强制使用交互式后端
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("'TkAgg' backend not found, trying default interactive backend...")
            matplotlib.use(matplotlib.get_backend())
            import matplotlib.pyplot as plt

        # --- 2. 验证输入并选择特征字典 (*** 由 data_source 控制 ***) ---
        if data_source == 'training':
            feature_dict = self.instance_feature
        elif data_source == 'testing':
            feature_dict = self.to_be_solve_ins_feature
        else:
            raise ValueError(f"Invalid data_source '{data_source}'. Must be 'training' or 'testing'.")

        if not feature_dict:
            warnings.warn(f"No features found for data_source '{data_source}'. Nothing to plot.")
            return

        # --- 3. 读取 JSON 并提取聚类信息 (*** 由 cluster_key 控制 ***) ---
        cluster_assignments_raw = None

        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)

            # 验证 cluster_key
            valid_keys = ['cluster_id_instances', 'apply_cluster_of_each_instance', 'match_cluster_of_each_instance']
            if cluster_key not in valid_keys:
                raise ValueError(f"Invalid cluster_key '{cluster_key}'. Must be one of {valid_keys}")

            # *** 使用 cluster_key 参数动态读取
            cluster_assignments_raw = data.get(cluster_key)

            if not cluster_assignments_raw:
                print(f"Error: Key '{cluster_key}' not found or is empty in {json_file_path}.")
                return

            # --- 统一数据结构 ---
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

        # --- 4. 分配颜色 (*** 逻辑已重写为 V4 稳定规则 ***) ---
        preset_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
            '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d',
            '#9edae5', '#a55194', '#393b79'
        ]
        color_map = {}
        legend_elements = []

        # 灰色用于所有“其他”键
        DEFAULT_OTHER_COLOR = '#7f7f7f'

        all_cluster_keys = sorted(cluster_assignments.keys())

        for cluster_key in all_cluster_keys:
            color = ''
            label = f'Cluster: {cluster_key}'

            if cluster_key == 'main':
                # 规则 1: 'main' 是 黑色
                color = 'black'
            elif cluster_key.isdigit():
                # 规则 2: 数字键使用其数字作为索引
                cluster_num = int(cluster_key)
                index = cluster_num % len(preset_colors)
                color = preset_colors[index]
            else:
                # 规则 3: 其他所有键 (none, brute_force_best, etc.) 都是灰色
                color = DEFAULT_OTHER_COLOR

            # 只有当颜色还未被分配时才添加图例（防止 'none' 和 'brute_force' 都被设为灰色时重复）
            if color not in color_map.values() or color == 'black':
                legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label))
            elif color == DEFAULT_OTHER_COLOR and not any(
                    l.get_label() == 'Cluster: Other (grey)' for l in legend_elements):
                # 为所有灰色键创建一个统一的图例
                legend_elements.append(Line2D([0], [0], color=color, lw=2, label='Cluster: Other (grey)'))

            color_map[cluster_key] = color

        legend_elements.sort(key=lambda x: x.get_label())
        # --- 5. 反转映射：{id: color} ---
        # (此部分逻辑不变)
        instance_to_color_map = {}
        for cluster_key, instance_list in cluster_assignments.items():
            color = color_map[cluster_key]
            for instance_id in instance_list:
                instance_to_color_map[instance_id] = color

        # --- 6. 开始绘图 (*** 已更新为线图 ***) ---
        fig, ax = plt.subplots(figsize=(12, 9))
        plot_any = False

        # 遍历我们选择的特征字典 (training or testing)
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

            # *** 修改为线图 (linestyle='-') ***
            ax.plot(x_coords, y_coords, color=plot_color, alpha=0.6, linestyle='-')

            # (可选) 如果你还想要落点圆圈，取消下面这行的注释
            # ax.plot(x_coords[-1], y_coords[-1], 'o', color=plot_color, markersize=4)

            # 标注ID
            ax.text(x_coords[-1], y_coords[-1] + 0.05, f'{instance_id}', color=plot_color,
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

        # --- 7. 设置图像美化和信息 (*** 标题已更新 ***) ---
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

        # 8. 添加图例
        if legend_elements:
            # ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0))
            # 方案 A: 让 matplotlib 自动寻找最佳空白位置 (推荐)
            ax.legend(handles=legend_elements, loc='best')

            # 方案 B: 如果想强制放在右上角，可以用这个：
            # ax.legend(handles=legend_elements, loc='upper right')

            # 方案 C: 如果图例条目太多，可以分列显示，避免太长挡住图
            # ax.legend(handles=legend_elements, loc='best', ncol=2, fontsize='small')

        if not plot_any:
            ax.text(0.5, 0.5,
                    f"No instances from '{data_source}' feature set\nwere found in the JSON cluster map (Key: {cluster_key}).",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')

        # plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为外部图例留出空间
        plt.tight_layout()

        # 9. 显示图像
        print(
            f"Displaying clustered plot window...\n  Data Source: '{data_source}'\n  Cluster Key: '{cluster_key}'\n(Close the window to continue the script)")
        plt.show()

        plt.close(fig)

    def filter_custom_instances_by_x(self, instance_input: list | dict, threshold: float = 4.5) -> tuple[list, list]:
        """
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

        # 1. 输入标准化
        if isinstance(instance_input, list):
            target_dict = {i: i for i in instance_input}
        elif isinstance(instance_input, dict):
            target_dict = instance_input
        else:
            raise ValueError("Input must be a list of seeds or a dict of {id: seed}.")

        print(f"🚀 Start filtering {len(target_dict)} custom instances (Threshold: +/-{threshold})...")

        satisfied_ids = []  # 存越界的 (极端)
        not_satisfied_ids = []  # 存正常的

        count = 0
        for inst_id, seed in target_dict.items():
            count += 1
            print(f"   -> Processing {count}/{len(target_dict)}...", end='\r')

            try:
                # 现场跑一遍环境获取特征
                feature = self.feature_pipeline(seed=seed)

                if not feature:
                    print(f"   -> ⚠️ Failed to generate feature for seed {seed}")
                    continue

                # 解析 X 坐标
                mid = len(feature) // 2
                x_coords = feature[:mid]

                # 判断逻辑
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

    # 全分布 的20 个实例
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

    # --- (新功能) 调用 show_clustered_features ---
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
