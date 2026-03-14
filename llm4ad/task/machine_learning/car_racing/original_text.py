import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 创建 CarRacing 环境
# env = gym.make('CarRacing-v3', render_mode='human')
env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)


def heuristic_action(observation):
    """
    更复杂的启发式策略：
    动作定义：
    [steering, gas, brake]
    """
    # 初始化动作
    action = np.array([0.0, 0.0, 0.0])  # [steering, gas, brake]

    # 提取白色赛道部分
    white_threshold = 200  # 白色的阈值
    white_mask = (observation[:, :, 0] > white_threshold) & \
                 (observation[:, :, 1] > white_threshold) & \
                 (observation[:, :, 2] > white_threshold)

    # 计算赛道中心线
    white_indices = np.argwhere(white_mask)
    if len(white_indices) > 0:
        center_x = np.mean(white_indices[:, 1])  # 赛道中心线的 x 坐标
    else:
        center_x = observation.shape[1] // 2  # 如果没有白色区域，默认中心线为中间

    # 计算赛车的位置相对于赛道中心线
    car_position = observation.shape[1] // 2  # 假设赛车在图像的中间位置
    offset = center_x - car_position

    # 根据偏移量调整转向和刹车
    if offset > 10:  # 偏离中心线向右
        action[0] = 0.5  # steer right
        action[1] = 0.0  # 不加油门
        action[2] = 0.2  # 轻踩刹车
    elif offset < -10:  # 偏离中心线向左
        action[0] = -0.5  # steer left
        action[1] = 0.0  # 不加油门
        action[2] = 0.2  # 轻踩刹车
    else:
        action[0] = 0.0  # 保持直行
        action[1] = 0.8  # 直行时加油门
        action[2] = 0.0  # 不刹车

    # 根据赛道宽度调整刹车
    white_density = np.sum(white_mask) / (observation.shape[0] * observation.shape[1])  # 白色区域密度
    if white_density < 0.1:  # 赛道窄，减速或刹车
        action[1] = 0.4  # 减速
        action[2] = 0.3  # 增加刹车力度

    return action

def heuristic_action_4o(observation):
    action = np.array([0.0, 0.0, 0.0])
    white_threshold = 200  # 白色的阈值
    white_mask = (observation[:, :, 0] > white_threshold) & \
                 (observation[:, :, 1] > white_threshold) & \
                 (observation[:, :, 2] > white_threshold)

    white_indices = np.argwhere(white_mask)
    center_x = np.mean(white_indices[:, 1]) if len(white_indices) > 0 else observation.shape[1] // 2
    car_position = observation.shape[1] // 2
    offset = center_x - car_position

    # 动态调整转向
    steering_angle = np.clip(offset / 100.0, -1.0, 1.0)  # 使用比例控制
    action[0] = steering_angle

    if abs(offset) > 10:
        action[1] = 0.0  # 不加油门
        action[2] = 0.2  # 轻踩刹车
    else:
        action[1] = 0.8  # 加油门
        action[2] = 0.0  # 不刹车

    # 根据赛道宽度调整刹车
    white_density = np.sum(white_mask) / (observation.shape[0] * observation.shape[1])
    if white_density < 0.1:
        action[1] = 0.4  # 减速
        action[2] = 0.3  # 增加刹车力度

    return action


def display_observation(observation):
    """
    显示当前的观察值。
    """
    plt.imshow(observation)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


# 重置环境
observation, _ = env.reset(seed=42)

done = False
step = 0
sum_reward = 0
while not done:
    # 获取启发式动作
    if step % 20 == 0:
        pre_observation = observation
    step += 1
    print(step)
    action = heuristic_action(observation)
    # action = heuristic_action_4o(observation)
    # 执行动作

    observation, reward, done, info, _ = env.step(action)
    sum_reward += reward
    print(step, sum_reward)

    # obs1_upper = observation[:83, :, :]
    obs1_upper = observation[:, :, :]

    if np.mean(pre_observation) == np.mean(observation):
        break
    # 显示当前的观察值
    if step > 200 and step % 30 == 0:
        display_observation(obs1_upper)

    # 渲染环境
    env.render()

# 关闭环境
env.close()
print('赛车终止')
display_observation(obs1_upper)
print(sum_reward)