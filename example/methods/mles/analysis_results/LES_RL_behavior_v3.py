import os
import json
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
import traceback


def read_evolution_trajectory_with_resets(path, sample_step, evals_per_step=4):
    """
    Modified version that also tracks environment resets
    Returns:
    - trajectory_sample: filled trajectory by sample step (original)
    - trajectory_reset: filled trajectory by environment reset count
    """
    json_path = os.path.join(path, 'samples', 'samples_best.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    sample_orders = [entry['sample_order'] for entry in data]
    scores = [entry['score'] for entry in data]

    # Original trajectory (by sample step)
    trajectory_sample = np.zeros(sample_step)

    # New trajectory (by environment resets)
    total_resets = sample_step * evals_per_step
    trajectory_reset = np.zeros(total_resets)

    if not sample_orders:
        return trajectory_sample, trajectory_reset

    # Fill both trajectories
    current_value = scores[0]
    current_idx = 0

    for i in range(sample_step):
        # Update current value if we've passed a sample order point
        if current_idx < len(sample_orders) and i >= sample_orders[current_idx]:
            current_value = scores[current_idx]
            current_idx += 1
        trajectory_sample[i] = current_value

        # Fill the corresponding reset positions
        reset_start = i * evals_per_step
        reset_end = (i + 1) * evals_per_step
        trajectory_reset[reset_start:reset_end] = current_value

    return trajectory_sample, trajectory_reset


def read_evolution_trajectory_RL(path, env_end, monotonic_increase=True):
    """
    Read RL path's training_data.json file and extract NWS metrics.
    Args:
        path: Path to RL results
        env_end: Length of trajectory to return
        monotonic_increase: If True, the trajectory will only record improvements (monotonically increasing)
    Returns:
        np.array: Trajectory of nws values
    """
    json_path = os.path.join(path, 'training_history.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    evaluation_history = data['evaluation_history']

    if not evaluation_history:
        return np.zeros(env_end)

    # Extract episodes and corresponding nws values
    episodes = []
    nws_values = []

    for entry in evaluation_history:
        episodes.append(entry['episode'])
        nws_values.append(entry['metrics']['nws'])

    # Create trajectory with interpolation
    trajectory = np.zeros(env_end)
    current_value = nws_values[0]
    current_idx = 0

    if monotonic_increase:
        # For monotonic increase, we keep track of the maximum value seen so far
        max_value = nws_values[0]
        max_idx = 0

        for i in range(env_end):
            # Check if we've reached a new evaluation point
            if current_idx < len(episodes) and i >= episodes[current_idx]:
                current_value = nws_values[current_idx]
                current_idx += 1
                # Update max value if current evaluation is better
                if current_value > max_value:
                    max_value = current_value
            trajectory[i] = max_value
    else:
        # Original behavior (non-monotonic)
        for i in range(env_end):
            if current_idx < len(episodes) and i >= episodes[current_idx]:
                current_value = nws_values[current_idx]
                current_idx += 1
            trajectory[i] = current_value

    return trajectory


def plot_trajectories(path_dict, sample_step, evals_per_step, by_reset, title, lower_bound=0):
    """Plot trajectories with a customizable lower bound."""
    plt.figure(figsize=(8, 6))

    for method_name, paths in path_dict.items():
        trajectories = []

        for path in paths:
            try:
                traj_sample, traj_reset = read_evolution_trajectory_with_resets(
                    path, sample_step, evals_per_step)
                trajectory = traj_reset if by_reset else traj_sample
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        if not trajectories:
            continue

        trajectories = np.array(trajectories)
        mean_traj = np.maximum(np.mean(trajectories, axis=0), lower_bound)
        std_traj = np.std(trajectories, axis=0)

        x = np.arange(len(mean_traj))
        plt.plot(x, mean_traj, label=method_name)

        plt.fill_between(
            x,
            np.maximum(mean_traj - std_traj, lower_bound),
            np.maximum(mean_traj + std_traj, lower_bound),
            alpha=0.2
        )

    plt.xlabel('Sample Step' if not by_reset else 'Environment Reset Count')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def analysis_method_performance(
        LES_path_dict,
        RL_method_root,
        sample_step,
        evals_per_step=4,
        rl_monotonic=True,
        lower_bound=0.0
):
    # 创建绘图上下文
    size = 20
    with plt.rc_context({
        'font.family': 'Times New Roman',
        'font.size': size,
        'axes.titlesize': size,
        'axes.labelsize': size,
        'xtick.labelsize': size,
        'ytick.labelsize': size,
        'legend.fontsize': size,
        'figure.titlesize': 18
    }):
        """Plot LES and RL trajectories with a customizable lower bound and print final stats."""
        fig = plt.figure(figsize=(11, 6))

        # 设置图像边框颜色
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(0.5)

        env_end = sample_step * evals_per_step

        # Dictionary to store results
        method_stats = {}

        # Define colors and styles
        method_styles = {
            'MLES': {'color': 'red', 'marker': 'o', 'linestyle': '-', 'linewidth': 3.0, 'markevery': 0.08,
                     'markersize': 7},
            'Eoh': {'color': 'darkorange', 'marker': '', 'linestyle': '-', 'linewidth': 1.75, 'markevery': 0.08,
                    'markersize': 5},
            'DQN': {'color': 'green', 'marker': '', 'linestyle': '--', 'linewidth': 1.75, 'markevery': 0.08,
                    'markersize': 5},
            'PPO': {'color': 'blue', 'marker': 'd', 'linestyle': '--', 'linewidth': 1.75, 'markevery': 0.08,
                    'markersize': 5}
        }

        # Process LES methods
        for method_name, paths in LES_path_dict.items():
            trajectories = []
            for path in paths:
                try:
                    traj_sample, traj_reset = read_evolution_trajectory_with_resets(
                        path, sample_step, evals_per_step)
                    trajectories.append(traj_reset)
                except Exception as e:
                    print(f"Error processing LES {path}: {e}")
                    continue

            if not trajectories:
                continue

            trajectories = np.array(trajectories)
            mean_traj = np.maximum(np.mean(trajectories, axis=0), lower_bound)
            final_means = trajectories[:, -1]
            final_best = np.max(final_means)
            final_mean = np.mean(final_means)


            n = len(final_means)
            variance = np.var(final_means)
            sem = np.std(final_means) / np.sqrt(n) if n > 1 else 0.0

            method_stats[f"LES-{method_name}"] = {
                'final_mean': final_mean,
                'final_best': final_best,
                'final_variance': variance,
                'final_sem': sem
            }

            std_traj = np.std(trajectories, axis=0)

            n = len(trajectories)
            sem_traj = np.std(trajectories, axis=0) / np.sqrt(n)  # SEM = STD / sqrt(n)

            x = np.arange(len(mean_traj))
            style = method_styles.get(method_name, {})

            if method_name == "MLES":
                lablename = 'MLES (Ours)'
            else:
                lablename = f"LES-{method_name}"
            plt.plot(x, mean_traj,
                     label=lablename,
                     color=style['color'],
                     linestyle=style['linestyle'],
                     marker=style['marker'],
                     markevery=style['markevery'],
                     markersize=style['markersize'],
                     linewidth=style['linewidth'])
            alpha = 0.1
            plt.fill_between(
                x,
                np.maximum(mean_traj - sem_traj, lower_bound),
                np.maximum(mean_traj + sem_traj, lower_bound),
                color=style['color'],
                alpha=alpha
            )

        # Process RL methods
        for method_name, paths in RL_method_root.items():
            trajectories = []
            for path in paths:
                try:
                    trajectory = read_evolution_trajectory_RL(path, env_end, monotonic_increase=rl_monotonic)
                    trajectories.append(trajectory)
                except Exception as e:
                    print(f"Error processing RL {path}: {e}")
                    continue

            if not trajectories:
                continue

            trajectories = np.array(trajectories)
            mean_traj = np.maximum(np.mean(trajectories, axis=0), lower_bound)
            ori_mean_traj = np.mean(trajectories, axis=0)
            final_means = trajectories[:, -1]
            final_best = np.max(final_means)
            final_mean = np.mean(final_means)


            n = len(final_means)
            variance = np.var(final_means)
            sem = np.std(final_means) / np.sqrt(n) if n > 1 else 0.0

            method_stats[f"RL-{method_name}"] = {
                'final_mean': final_mean,
                'final_best': final_best,
                'final_variance': variance,
                'final_sem': sem
            }

            std_traj = np.std(trajectories, axis=0)

            n = len(trajectories)
            sem_traj = np.std(trajectories, axis=0) / np.sqrt(n)  # SEM = STD / sqrt(n)

            x = np.arange(len(mean_traj))
            line_label = f"DRL-{method_name}"
            # if rl_monotonic:
            #     line_label += " (monotonic)"

            style = method_styles.get(method_name, {})
            plt.plot(x, mean_traj,
                     label=line_label,
                     color=style['color'],
                     linestyle=style['linestyle'],
                     marker=style['marker'],
                     markevery=style['markevery'],
                     markersize=style['markersize'],
                     linewidth=style['linewidth']
                     )
            plt.fill_between(
                x,
                np.maximum(ori_mean_traj - sem_traj, lower_bound),
                np.maximum(ori_mean_traj + sem_traj, lower_bound),
                color=style['color'],
                alpha=alpha
            )


        print("\nMethod Performance Statistics:")
        print("{:<20} {:<15} {:<15} {:<15} {:<15}".format(
            "Method", "Final Mean", "Final SEM", "Final Best", "Final Variance"))
        print("-" * 80)
        for method, stats in method_stats.items():
            print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                method,
                stats['final_mean'],
                stats['final_sem'],
                stats['final_best'],
                stats['final_variance']))

        plt.xlabel('Environment reset count', fontname='Times New Roman')
        plt.ylabel('Quantitative metric (NWS)', fontname='Times New Roman')


        legend = plt.legend()
        for text in legend.get_texts():
            text.set_fontname('Times New Roman')

        plt.xlim(-100, 10100)
        plt.grid(True,
                 color='#C0C0C0',
                 linestyle='--',
                 linewidth=0.5,
                 alpha=0.8
                 )
        plt.tight_layout()


        file_name = "moonlander_evolurion_process_v2.pdf"
        dpi_value = 600

        plt.savefig(file_name, dpi=dpi_value, bbox_inches='tight')
        print(f"Image save: {file_name}，DPI: {dpi_value}")
        plt.show()


if __name__ == "__main__":

    tasks = ['lunar', 'car']
    task = tasks[0]
    if task == 'lunar':
        # Lunar lander
        LES_method_root = {
            'MLES': [
                r'..\example\moon_lander\batch\mmEoh\v0526_0\20250526_213816_Problem_EoH',
                r'..\example\moon_lander\batch\mmEoh\v0526_2\20250526_233234_Problem_EoH',
                r'..\example\moon_lander\batch\mmEoh\v0526_4\20250528_011828_Problem_EoH',
                r'..\example\moon_lander\batch\mmEoh\v0526_6\20250604_210831_Problem_EoH',
                r'..\example\moon_lander\batch\mmEoh\v0526_8\20250605_010332_Problem_EoH',
            ],
            'Eoh': [
                r'..\example\moon_lander\batch\Eoh\v0526_1\20250527_104314_Problem_EoH',
                r'..\example\moon_lander\batch\Eoh\v0526_1\20250527_175911_Problem_EoH',
                r'..\example\moon_lander\batch\Eoh\v0526_2\20250528_131828_Problem_EoH',
                r'..\example\moon_lander\batch\Eoh\v0526_3\20250528_214635_Problem_EoH',
                r'..\example\moon_lander\batch\Eoh\v0526_3\20250528_214637_Problem_EoH'

            ]
        }

        RL_method_root = {
            'DQN': [
                r'..\..\rl_experiment\results\dqn_lunarlander_20250603_003755',
                r'..\..\rl_experiment\results\dqn_lunarlander_20250603_003756',
                r'..\..\rl_experiment\results\dqn_lunarlander_20250603_022211',
                r'..\..\rl_experiment\results\dqn_lunarlander_20250603_024024',
                r'..\..\rl_experiment\results\dqn_lunarlander_20250603_152545',
            ],
            'PPO': [
                r"..\..\rl_experiment\results\ppo_lunarlander_20250603_094751",
                r'..\..\rl_experiment\results\ppo_lunarlander_20250603_095146',
                r'..\..\rl_experiment\results\ppo_lunarlander_20250603_112742',
                r'..\..\rl_experiment\results\ppo_lunarlander_20250603_112848',
                r'..\..\rl_experiment\results\ppo_lunarlander_20250603_130517',
            ]
        }

        analysis_method_performance(LES_method_root, RL_method_root, sample_step=2000, rl_monotonic=True,
                                    evals_per_step=5,
                                    lower_bound=0.5)

    else:
        # Car Racing
        LES_method_root = {
            'MLES': [
                r'..\example\racingcar\batch\mmEoh\v0526_1\20250527_010943_Problem_EoH',
                r'..\example\racingcar\batch\mmEoh\v0526_8\20250529_035806_Problem_EoH',
                r'..\example\racingcar\batch\mmEoh\v0526_2\20250527_010943_Problem_EoH',
                r'..\example\racingcar\batch\mmEoh\v0526_22\20250607_170916_Problem_EoH',
                r'..\example\racingcar\batch\mmEoh\v0526_10\20250605_112400_Problem_EoH'
            ],
            'Eoh': [
                r'..\example\racingcar\All\Eoh\v0526_0\20250526_174026_Problem_EoH',
                r'..\example\racingcar\batch\Eoh\v0526_0\20250529_092724_Problem_EoH',
                r'..\example\racingcar\batch\Eoh\v0526_1\20250529_175413_Problem_EoH',
                r'..\example\racingcar\batch\Eoh\v0526_1\20250529_175544_Problem_EoH',
                r'..\example\racingcar\batch\Eoh\v0526_2\20250529_232855_Problem_EoH'
            ]
        }

        RL_method_root = {
            'DQN': [
                r'..\..\rl_experiment\results\dqn_carracing_20250605_115728_simple',
                r'..\..\rl_experiment\results\dqn_carracing_20250608_115712_simple',
                r'..\..\rl_experiment\results\dqn_carracing_20250610_144911_simple',
                r'..\..\rl_experiment\results\dqn_carracing_20250613_122801_simple',
                r'..\..\rl_experiment\results\dqn_carracing_20250615_053724_simple',
            ],
            'PPO': [
                r'..\..\rl_experiment\results\ppo_carracing_continuous_20250531_002542_simple',
                r'..\..\rl_experiment\results\ppo_carracing_continuous_20250605_011022',
                r'..\..\rl_experiment\results\ppo_carracing_continuous_20250606_223600',
                r'..\..\rl_experiment\results\ppo_carracing_continuous_20250617_010440_simple',
                r'..\..\rl_experiment\results\ppo_carracing_continuous_20250618_052952_simple',]
        }

        analysis_method_performance(LES_method_root, RL_method_root, sample_step=2000, rl_monotonic=True,
                                    evals_per_step=4,
                                    lower_bound=0.0)
