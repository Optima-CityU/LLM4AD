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

from typing import Optional, Tuple, List, Any
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
        self.non_image_representation_explanation = non_image_representation_explanation

        self.gravity = kwargs.get('gravity', -10.0)
        self.enable_wind = kwargs.get('enable_wind', False)
        self.wind_power = kwargs.get('wind_power', 15.0)
        self.turbulence_power = kwargs.get('turbulence_power', 1.5)

    def create_base64(self, original_input, fitness, recoder, which_image):
        """
        这个函数将图片变成base64的形式，最终输出base64
        """
        img_bytes = io.BytesIO()
        plt.imshow(original_input.astype(np.uint8))
        image_recode = recoder[f'{which_image}']

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

    def evaluate(self, action_select: callable, env_seeds=[42, 520, 1231, 114, 886]) -> Optional[dict]:
        try:
            total_rewards = []
            total_fuel = 0
            image64s = []
            observations = []
            success_count = 0
            num_episodes = len(env_seeds)
            episodes_recorder = {}
            for i in range(num_episodes):
                each_evaluate_result = self.evaluate_single(action_select, env_seed=env_seeds[i])
                if each_evaluate_result is not None:
                    infos = each_evaluate_result[0]
                    total_rewards.append(infos['episode_reward'])
                    total_fuel += infos['episode_fuel']
                    image64s.append(each_evaluate_result[1])
                    observations.append(infos['observations'])
                    if infos['episode_reward'] >= 200:
                        success_count += 1
                    episodes_recorder[f'{i}'] = infos

            mean_reward = np.mean(total_rewards)
            mean_fuel = total_fuel / num_episodes
            success_rate = success_count / num_episodes

            which_image = total_rewards.index(min(total_rewards))
            chosen_image = image64s[which_image]  # 默认选得分最低的一张
            observation_chosen = observations[which_image]

            # 标准化加权得分（权重α=0.6, β=0.2, γ=0.2）
            nws = (mean_reward / 200) * 0.6 + (1 - min(mean_fuel / 100, 1)) * 0.2 + success_rate * 0.2
            encoded_base64 = self.create_base64(chosen_image, nws, episodes_recorder, which_image)
            observation_chosen_str = str(observation_chosen)

            test_result = {
                'Mean Reward': mean_reward,
                'Mean Fuel': mean_fuel,
                'Success Rate': success_rate,
                'NWS': nws
            }
            if self.whocall == 'mles':
                return {'score': nws, 'image': encoded_base64, 'observation': observation_chosen_str,
                        'Test result': test_result}
            else:
                return nws
        except Exception as e:
            print(e)
            traceback.print_exc()  # 打印完整的异常堆栈信息
            return None

    def merge_evaluate(self, action_selects: List[callable], env_seeds=[42, 520, 1231, 114, 886]) -> Optional[dict]:
        try:
            total_rewards = []
            total_fuel = 0
            image64s = []
            observations = []
            success_count = 0
            num_episodes = len(env_seeds)
            episodes_recorder = {}
            for i in range(num_episodes):
                each_evaluate_result = self.evaluate_single_merge(action_selects, env_seed=env_seeds[i])
                if each_evaluate_result is not None:
                    infos = each_evaluate_result[0]
                    total_rewards.append(infos['episode_reward'])
                    total_fuel += infos['episode_fuel']
                    image64s.append(each_evaluate_result[1])
                    observations.append(infos['observations'])
                    if infos['episode_reward'] >= 200:
                        success_count += 1
                    episodes_recorder[f'{i}'] = infos

            mean_reward = np.mean(total_rewards)
            mean_fuel = total_fuel / num_episodes
            success_rate = success_count / num_episodes

            which_image = total_rewards.index(min(total_rewards))
            chosen_image = image64s[which_image]  # 默认选得分最低的一张
            observation_chosen = observations[which_image]

            # 标准化加权得分（权重α=0.6, β=0.2, γ=0.2）
            nws = (mean_reward / 200) * 0.6 + (1 - min(mean_fuel / 100, 1)) * 0.2 + success_rate * 0.2
            encoded_base64 = self.create_base64(chosen_image, nws, episodes_recorder, which_image)
            observation_chosen_str = str(observation_chosen)

            test_result = {
                'Mean Reward': mean_reward,
                'Mean Fuel': mean_fuel,
                'Success Rate': success_rate,
                'NWS': nws
            }
            if self.whocall == 'mles':
                return {'score': nws, 'image': encoded_base64, 'observation': observation_chosen_str,
                        'Test result': test_result}
            else:
                return nws
        except Exception as e:
            print(e)
            traceback.print_exc()  # 打印完整的异常堆栈信息
            return None

    def evaluate_single_merge(self, action_selects: List[callable], env_seed=42):
        """Evaluate heuristic function on moon lander problem."""
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
            # Collect votes from all policies
            votes = []
            for policy in action_selects:
                voted_action = policy(observation, action, pre_observation)
                votes.append(voted_action)

            # Implement voting mechanism (majority vote)
            action_counts = {}
            for vote in votes:
                action_counts[vote] = action_counts.get(vote, 0) + 1

            # Select action with most votes
            action = max(action_counts.items(), key=lambda x: x[1])[0]

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
                infos = {'done': done,
                         'truncated': truncated,
                         'episode_fuel': episode_fuel,
                         'episode_reward': episode_reward,
                         'observations': observations}
                return infos, canvas

    def evaluate_single(self, action_select: callable, env_seed=42):
        """Evaluate heuristic function on moon lander problem."""
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
                infos = {'done': done,
                         'truncated': truncated,
                         'episode_fuel': episode_fuel,
                         'episode_reward': episode_reward,
                         'observations': observations}
                return infos, canvas

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)
