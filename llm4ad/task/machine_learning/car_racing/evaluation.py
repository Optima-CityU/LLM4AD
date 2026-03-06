
from __future__ import annotations

from typing import Optional, Tuple, List, Any, Set
import gymnasium as gym
import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.machine_learning.car_racing.template import template_program, task_description

import traceback
import matplotlib

matplotlib.use('Agg')  # 选择不显示的后端
import matplotlib.pyplot as plt
import io
from io import BytesIO
import base64
import copy
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

import warnings
import time

__all__ = ['RacingCarEvaluation']


# def evaluate(env: gym.Env, action_select: callable) -> float | None:


class RacingCarEvaluation(Evaluation):
    """Evaluator for moon lander problem."""

    def __init__(self, whocall='Eoh', max_steps=1200, timeout_seconds=180, **kwargs):
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

        self.env_name = "CarRacing-v3"
        self.env_max_episode_steps = max_steps
        self.whocall = whocall
        objective_value = kwargs.get('objective_value', 230)
        self.final_objective_score = objective_value

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

        self.env_mode = kwargs.get("env_mode", 'rgb_array')  # 从 kwargs 中安全提取 env_mode，默认值为 rgb_array

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

        num_episodes = len(ins_to_be_evaluated_id)
        episodes_recorder = {}

        for ins_id in ins_to_be_evaluated_id:
            env_seed = ins_to_be_evaluated_set[ins_id]

            each_evaluate_result = self.evaluate_single(action_select, env_seed=env_seed, skip_frame=1)

            if each_evaluate_result is not None:
                infos = each_evaluate_result[0]

                total_rewards[ins_id] = infos['track_coverage']
                image64s[ins_id] = each_evaluate_result[1]

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


        min_reward_id = min(total_rewards, key=total_rewards.get)
        chosen_image_base64 = image64s[min_reward_id]  # 默认选得分最低的一张

        sorted_keys = sorted(instance_performance.keys())
        # 返回按 key 排序的 score 列表
        list_performance = [instance_performance[k]['score'] for k in sorted_keys]

        if self.whocall == 'mles':
            return {'score': mean_reward,
                    'image': chosen_image_base64,
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
        """Evaluate heuristic function on racing car problem."""
        env = gym.make(self.env_name, render_mode=self.env_mode, domain_randomize=False, continuous=True)  # 'rgb_array'
        observation, _ = env.reset(seed=env_seed)  # initialization
        start_time = time.time()
        action = np.array([0.0, 1.0, 0.0])  # initial action
        episode_reward = 0
        episode_max_reward = 0  # 新增：用于记录过程中最高累计 reward

        # --- 记录轨迹和视野范围 ---
        trajectory = []
        car_angles = []  # 新增：记录车辆角度
        view_rectangles = []  # 存储(中心x, 中心y, 角度, 长度, 宽度)
        done = False
        # 视角参数配置
        view_length = 46.0  # 视野长度
        view_width = 38.0  # 视野宽度
        view_offset = 14.0  # 视野中心在车头前方的偏移量

        pre_observation = copy.deepcopy(observation)
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        step = 0
        while not done and step < self.env_max_episode_steps:
            car_velocity = env.unwrapped.car.hull.linearVelocity
            speed = np.sqrt(car_velocity[0] ** 2 + car_velocity[1] ** 2)  # 计算速度大小
            action = action_select(observation,
                                   speed,
                                   action,
                                   pre_observation)
            pre_observation = copy.deepcopy(observation)

            for _ in range(skip_frame):
                observation, reward, done, truncated, info = env.step(action)
                step += 1

                # 记录车辆位置和角度
                car_pos = env.unwrapped.car.hull.position
                car_angle = env.unwrapped.car.hull.angle  # 车辆当前角度(弧度)

                trajectory.append((car_pos.x, car_pos.y))
                car_angles.append(car_angle)  # 记录角度

                # 修正视角偏移计算（考虑Box2D角度定义）
                # Box2D中0弧度通常指向右方，我们需要调整为指向上方
                corrected_angle = car_angle + np.pi / 2  # 旋转90度

                # 计算视野中心位置(车头前方偏移)
                view_center_x = car_pos.x + np.cos(corrected_angle) * view_offset
                view_center_y = car_pos.y + np.sin(corrected_angle) * view_offset

                view_rectangles.append((view_center_x, view_center_y, corrected_angle, view_width, view_length))

                episode_reward += reward
                episode_max_reward = max(episode_max_reward, episode_reward)  # 更新最高值

        # --- 第三部分：绘制完整轨迹和视野范围 ---
        plt.figure(figsize=(9, 8))
        green_color = '#62f972'
        plt.gca().set_facecolor(green_color)

        # 重新绘制赛道 - 这次不添加label，后面统一添加图例
        # for polygon in env.unwrapped.road_poly:
        #     vertices = polygon[0]
        #     color = polygon[1]
        #     if hasattr(color, '__iter__') and not isinstance(color, tuple):
        #         color = tuple(color)
        #     fill_color = '#FFFFFF'
        #     if isinstance(color, tuple) and len(color) == 3:
        #         if color == (255, 255, 255):
        #             fill_color = '#FFFFFF'
        #         elif color == (255, 0, 0):
        #             fill_color = '#FF0000'
        #     x_coords = [v[0] for v in vertices] + [vertices[0][0]]
        #     y_coords = [v[1] for v in vertices] + [vertices[0][1]]
        #     plt.fill(x_coords, y_coords, color=fill_color, alpha=1.0)
        for polygon in env.unwrapped.road_poly:
            vertices = polygon[0]
            color = polygon[1]

            # 转换可能的numpy数组为元组
            if hasattr(color, '__iter__') and not isinstance(color, tuple):
                color = tuple(color)

            # 默认颜色处理
            fill_color = '#666666'  # 基础道路颜色 (102,102,102)

            # 处理带有颜色变化的情况
            if isinstance(color, tuple) and len(color) == 3:
                # 将颜色分量转换为整数（处理可能的浮点值）
                r = max(0, min(255, int(round(color[0]))))
                g = max(0, min(255, int(round(color[1]))))
                b = max(0, min(255, int(round(color[2]))))

                # 生成HEX颜色代码
                fill_color = "#{:02X}{:02X}{:02X}".format(r, g, b)

            # 闭合多边形坐标
            x_coords = [v[0] for v in vertices] + [vertices[0][0]]
            y_coords = [v[1] for v in vertices] + [vertices[0][1]]

            plt.fill(x_coords, y_coords, color=fill_color, alpha=1.0)

        view_color = '#8000FF'
        arrow_interval = 40 # 每20个点加一个箭头
        # 绘制所有视野范围(带旋转和偏移)
        for idy, rect in enumerate(view_rectangles):
            if idy == 0 or idy == len(view_rectangles) - 1 or idy % arrow_interval == 0:
                center_x, center_y, angle, length, width = rect

                # 创建旋转矩形(无边框)
                rect_patch = patches.Rectangle(
                    (-length / 2, -width / 2),  # 左下角相对于中心
                    length,
                    width,
                    linewidth=0,
                    edgecolor='none',
                    facecolor=view_color,
                    alpha=0.1
                )

                # 应用旋转和平移变换
                t = Affine2D().rotate(angle).translate(center_x, center_y) + plt.gca().transData
                rect_patch.set_transform(t)
                plt.gca().add_patch(rect_patch)

        arrow_color = '#FF6A00'
        # 绘制轨迹
        if trajectory:
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-', color='#FFD700', linewidth=1, label='Trajectory')
            # plt.scatter(trajectory[0, 0], trajectory[0, 1], c='#1E90FF', s=100, label='Start Point')
            # plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='#FF00FF', s=100, label='End Point')

            # 添加箭头表示车头方向

            for i in range(len(trajectory)):
                # 始终绘制第一个和最后一个点的箭头
                if i == 0 or i == len(trajectory) - 1 or i % arrow_interval == 0:
                    x, y = trajectory[i, 0], trajectory[i, 1]
                    angle = car_angles[i] + np.pi / 2  # 修正角度，与前面一致
                    dx = np.cos(angle) * 3  # 箭头长度
                    dy = np.sin(angle) * 5

                    # 调整箭头位置，使其从轨迹线上开始
                    arrow_start_x = x - dx * 0.3
                    arrow_start_y = y - dy * 0.3

                    plt.arrow(arrow_start_x, arrow_start_y, dx, dy,
                              head_width=3, head_length=4, fc=arrow_color, ec=arrow_color)

        # 创建统一的图例项
        grass_patch = patches.Patch(color=green_color, label='Off-Track Area (Grass)')
        track_patch = patches.Patch(color='#666666', label='Track')
        border_patch = patches.Patch(color='red', label='Curbing (red-white pattern at sharp turns)')
        view_patch = patches.Patch(color=view_color, alpha=0.1, label="Agent's Dynamic Visual Field")

        # 获取自动生成的图例项（轨迹、起点、终点）
        handles, labels = plt.gca().get_legend_handles_labels()

        custom_handles = [grass_patch, track_patch, border_patch, view_patch]
        all_handles = custom_handles + handles

        # 去重
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

        # 2. 保存到缓冲区
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')  # 也可以选择 "jpg", "svg" 等
        buffer.seek(0)  # 回到缓冲区开头

        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        plt.close()  # 关闭图形，避免内存泄漏
        env.close()
        end_time = time.time()
        infos = {'done': done,
                 'truncated': truncated,
                 'episode_reward': episode_reward,
                 'track_coverage': track_coverage,
                 'episode_max_reward': episode_max_reward,
                'evaluate_time': end_time - start_time}
        return infos, img_base64

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        ins_to_be_evaluated_id = kwargs.get('ins_to_be_evaluated_id', None)
        training_mode = kwargs.get('training_mode', True)
        return self.evaluate(callable_func, ins_to_be_evaluated_id, training_mode)
