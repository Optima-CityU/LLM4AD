import os
import json
import numpy as np
import matplotlib.pyplot as plt


def read_evolution_trajectory(path, sample_step):
    # ... (保持你现有的读取逻辑，已经包含了取负数 scores = [-entry['score'] for entry in data])
    json_path = os.path.join(path, 'samples', 'samples_best.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    sample_orders = [entry['sample_order'] for entry in data]
    scores = [entry['score'] for entry in data]

    trajectory = np.zeros(sample_step)
    if not sample_orders: return trajectory

    current_value = scores[0]
    current_idx = 0
    for i in range(sample_step):
        if current_idx < len(sample_orders) and i >= sample_orders[current_idx]:
            current_value = scores[current_idx]
            current_idx += 1
        trajectory[i] = current_value
    return trajectory


def analysis_method_performance(path_dict, sample_step, font_size=20):
    """
    美化后的绘图函数
    :param font_size: 统一控制字体大小
    """
    # 1. 设置全局字体为 Times New Roman
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # 使用学术常用的配色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (method_name, paths) in enumerate(path_dict.items()):
        trajectories = []
        for path in paths:
            try:
                trajectory = read_evolution_trajectory(path, sample_step)
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        if not trajectories: continue

        trajectories = np.array(trajectories)
        mean_traj = np.mean(trajectories, axis=0)
        std_traj = np.std(trajectories, axis=0)

        x = np.arange(sample_step)
        color = colors[idx % len(colors)]

        # 3. 绘制均值实线
        ax.plot(x, mean_traj, label=method_name, color=color, linewidth=2)

        # 4. 绘制标准差阴影
        ax.fill_between(x,
                        mean_traj - std_traj,
                        mean_traj + std_traj,
                        color=color,
                        alpha=0.15,
                        edgecolor='none')

    # 5. 细节美化
    ax.set_xlabel('Sample Order', fontsize=font_size, fontweight='bold')
    ax.set_ylabel('Score (-Value)', fontsize=font_size, fontweight='bold')
    ax.set_title('Method Performance Comparison', fontsize=font_size + 2, fontweight='bold')

    # 刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=font_size - 2)

    # 图例设置
    ax.legend(loc='best', fontsize=font_size - 2, frameon=True, framealpha=0.9, edgecolor='gray')
    # 网格线优化：使用虚线，并置于底层
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

    # 去除上方和右方的边框 (可选，让图看起来更现代)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    group0 = {
        # 'EoH': [
        #     r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\Eoh\0\20260203_111451_Problem_EoH',
        #     ],
        'MLES': [
            r'C:\0_QL_work\014_mmeoh\LLM4AD_MLES\LLM4AD\example\mles_moonlander\logs\MLES\20260208_215147'],
        # 'ReEvo': [
        #     r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\Reevo\0\20260203_175052_Problem_Method']

    }
    group1 = {
        'EoH': [
            r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\Eoh\1\20260203_095229_Problem_EoH',
        ],
        'MEoH': [
            r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\MEoh\1\20260203_095233_Problem_EoH'],
        'ReEvo': [
            r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\Reevo\1\20260204_004000_Problem_Method']

    }
    group8 = {
        'EoH': [
            r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\Eoh\8\20260202_232340_Problem_EoH',
        ],
        'MEoH': [
            r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\MEoh\8\20260202_231056_Problem_EoH'],
        'ReEvo': [
            r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\Reevo\8\20260203_175113_Problem_Method']

    }
    group9 = {
        'EoH': [
            r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\Eoh\9\20260203_014353_Problem_EoH',
        ],
        'MEoH': [
            r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\MEoh\9\20260203_014357_Problem_EoH'],
        'ReEvo': [
            r'C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\Reevo\9\20260204_004022_Problem_Method']

    }

    groups = [group0, group1, group8, group9]
    group_selected = groups[0]

    analysis_method_performance(group_selected, sample_step=1000)
