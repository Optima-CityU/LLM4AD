import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO
import base64
import copy
import time


def moon_lander_feature(seed, env_max_episode_steps=200):
    """Evaluate heuristic function on moon lander problem."""
    start_time = time.time()
    env = gym.make('LunarLander-v3', render_mode='rgb_array',
                   gravity=-10,
                   enable_wind=False,
                   wind_power=15,
                   turbulence_power=1.5)
    observation, _ = env.reset(seed=seed)  # initialization
    action = 0  # initial action
    episode_reward = 0
    episode_fuel = 0

    canvas = np.zeros((400, 600, 3), dtype=np.float32)
    observations = []

    pre_observation = copy.deepcopy(observation)
    observation, reward, done, truncated, info = env.step(action)

    flash_calculator = 0
    for i in range(env_max_episode_steps + 1):  # protect upper limits
        action = 0
        pre_observation = copy.deepcopy(observation)
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        if action in [1, 2, 3]:
            episode_fuel += 1

        if flash_calculator >= 5:
            img = env.render()
            # 提取非黑色部分
            mask = np.any(img != [0, 0, 0], axis=-1)
            # 计算动态透明度

            alpha = i / env_max_episode_steps  # 假设最大步数为200，可以根据实际情况调整
            alpha = min(alpha, 1.0)  # 确保透明度不超过1
            # 将当前帧的非黑色部分叠加到画布上
            canvas[mask] = canvas[mask] * (1 - alpha) + img[mask] * alpha
            observations.append(observation)
            flash_calculator = 0
        flash_calculator += 1

    observations = np.array(observations)

    env.close()
    end_time = time.time()
    feature_x = observations[:, 0]
    feature_y = observations[:, 1]
    feature = np.concatenate((feature_x, feature_y))

    return feature


if __name__ == "__main__":
    feature = moon_lander_feature(42)
    print('finish')
