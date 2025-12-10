#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import glob
import logging
import subprocess
import concurrent.futures
from typing import List, Optional

# --- [!] 请您在此处配置 ---
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# 1. 定义可执行文件（methods）的完整路径
METHODS_DIR = os.path.join(SCRIPT_DIR, "bin")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "remote_results")

# 2. 定义实例文件（instances）所在的文件夹
INSTANCES_DIR = os.path.join(SCRIPT_DIR, "XLTEST")

# 3. 定义日志文件
LOG_FILE = os.path.join(SCRIPT_DIR, "log", "experiment_log.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# 默认值：5天 (如果命令行未指定)
DEFAULT_TIME_LIMIT = 5 * 24 * 3600  # 单位为秒

MAX_WORKERS = None


# --- 配置结束 ---


def setup_logging():
    """配置日志系统，同时输出到文件和控制台"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),  # 写入文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )


# --- [!] 修改点 1: get_command_args 接收 time_limit 参数 ---
def get_command_args(method_path: str, instance_path: str, time_limit: int) -> Optional[List[str]]:
    """
    根据传入的 method_path，返回完整的命令行参数列表。

    参数:
        method_path (str): 可执行文件的完整路径
        instance_path (str): 实例文件的完整路径
        time_limit (int): 实验的时长限制 (秒)

    返回:
        List[str]: 一个字符串列表
        Optional[None]: 如果此方法未定义命令，返回None，将跳过执行
    """

    instance_name = os.path.basename(instance_path)

    # 获取可执行文件的基本名称，用于判断
    method_name = os.path.basename(method_path)

    # --- [!] 请您在此处配置 ---
    #
    # 注意：请将所有硬编码的 TIME_LIMIT 替换为传入的 time_limit 变量
    #

    # 使用传入的 time_limit 变量
    limit_str = f"{time_limit}"

    if 'AILSII' in method_name:
        return ["java", "-jar", "-Xms2000m", "-Xmx4000m", f"bin/{method_name}", "-file", instance_path, "-rounded",
                "true", "-stoppingCriterion", "Time", "-limit", limit_str, "-output", f"remote_results/{method_name}"]

    elif method_name == "filo2":
        return [f"{method_path}", instance_path, "-t", limit_str, "--outpath", f"remote_results/{method_name}"]

    elif method_name == "hgs":
        return [f"{method_path}", instance_path, f"remote_results/{method_name}/{instance_name}.sol", "-seed", "1",
                "-t", limit_str]

    elif method_name == "hgs-TV":
        return [f"{method_path}", instance_path, "-sol", f"remote_results/{method_name}/{instance_name}.sol", "-seed",
                "1", "-t", limit_str, "-type", "Uchoa", "-bss", f"remote_results/{method_name}/{instance_name}.csv",
                "-deco", "BarycentreClustering"]

    else:
        # 其他未知的可执行文件
        logging.error(f"未知的可执行文件: {method_name}")
        return None

    # --- 配置结束 ---


# --- [!] 修改点 2: run_single_experiment 接收 time_limit 参数 ---
def run_single_experiment(task_info: tuple) -> None:
    """
    运行单个实验（一个method + 一个instance）。

    task_info 是一个元组，格式为 (method_path, instance_path, time_limit)。
    """
    method_path, instance_path, time_limit = task_info
    method_name = os.path.basename(method_path)
    instance_name = os.path.basename(instance_path)

    log_prefix = f"[{method_name} | {instance_name}]"

    try:
        # 1. 获取命令列表 (现在需要传递 time_limit)
        command_list = get_command_args(method_path, instance_path, time_limit)

        if command_list is None:
            return

        logging.info(f"{log_prefix} 开始运行 ({time_limit}秒限制)。命令: {' '.join(command_list)}")

        # --- 核心修改点 (与原脚本逻辑一致，但使用 time_limit 变量作为参数) ---

        if method_name == "AILSII_CPU.jar":

            # --- 分支 1: AILSII_CPU.jar (重定向到 .csv 文件) ---

            output_dir = os.path.join(SCRIPT_DIR, "remote_results", method_name)
            output_csv_path = os.path.join(output_dir, f"{instance_name}.csv")
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                logging.error(f"{log_prefix} 无法创建输出目录 {output_dir}: {e}")
                return

            logging.info(f"{log_prefix} 输出将重定向到: {output_csv_path}")

            with open(output_csv_path, 'w') as output_file:
                subprocess.run(
                    command_list,
                    stdout=output_file,  # 将 stdout 导入文件
                    stderr=subprocess.PIPE,  # 仍然捕获 stderr
                    text=True,
                    check=True,
                )

            logging.info(f"{log_prefix} 运行成功。结果已保存到 {output_csv_path}")

        else:

            # --- 分支 2: 所有其他方法 (捕获到内存) ---

            result = subprocess.run(
                command_list,
                capture_output=True,  # 捕获 stdout 和 stderr
                text=True,
                check=True,
            )

            stdout_snippet = result.stdout.strip()[:200].replace('\n', ' ')
            logging.info(f"{log_prefix} 运行成功。STDOUT(snippet): {stdout_snippet}...")

        # --- 异常处理 ---

    except subprocess.CalledProcessError as e:
        logging.error(f"{log_prefix} 运行失败。返回码: {e.returncode}")
        if e.stderr:
            logging.error(f"{log_prefix} STDERR: {e.stderr.strip()}")

    except FileNotFoundError:
        logging.error(f"{log_prefix} 运行失败。命令或文件未找到: {command_list[0]}")

    except Exception as e:
        logging.error(f"{log_prefix} 发生意外的脚本错误: {e}")


# --- [!] 修改点 3: main 函数接收 time_limit 参数并将其添加到 task 元组中 ---
def main(method_name: str = None, start_idx: Optional[int] = None, end_idx: Optional[int] = None,
         time_limit: int = DEFAULT_TIME_LIMIT):
    """主函数，负责编排所有任务"""
    METHOD_TO_RUN = os.path.join(METHODS_DIR, method_name) if method_name else None
    METHODS_OUTPUT_DIRS = os.path.join(OUTPUT_DIR, method_name)
    if not os.path.exists(METHODS_OUTPUT_DIRS):
        os.makedirs(METHODS_OUTPUT_DIRS, exist_ok=True)

    setup_logging()
    logging.info("--- 脚本开始运行 ---")
    logging.info(f"配置：时间限制设置为 {time_limit} 秒 ({time_limit / 3600:.2f} 小时)。")

    # 1. 查找所有实例文件
    instance_pattern = os.path.join(INSTANCES_DIR, "*.vrp")
    all_instance_files = glob.glob(instance_pattern)

    if not METHOD_TO_RUN:
        logging.critical("配置错误：METHOD_TO_RUN 为空。请检查脚本配置。")
        return

    if not all_instance_files:
        logging.critical(f"配置错误：在 {INSTANCES_DIR} 中未找到任何 *.vrp 文件。")
        return

    # 确定文件列表的子集
    all_instance_files.sort()

    start = start_idx if start_idx is not None else 0
    end = end_idx if end_idx is not None else len(all_instance_files)

    if start < 0 or start >= len(all_instance_files):
        start = 0
    if end > len(all_instance_files):
        end = len(all_instance_files)
    if start >= end:
        logging.critical(f"切片范围无效: start_idx ({start}) 必须小于 end_idx ({end})。")
        return

    instance_files = all_instance_files[start:end]

    logging.info(f"找到 1 个可执行文件 ({method_name})。")
    logging.info(
        f"在 {len(all_instance_files)} 个实例中，选择了从索引 {start} 到 {end - 1} (共 {len(instance_files)} 个文件) 运行。")

    # 2. 创建所有任务组合 (笛卡尔积)
    tasks = []
    for instance in instance_files:
        # [!] 关键修改：将 time_limit 添加到任务元组中
        tasks.append((METHOD_TO_RUN, instance, time_limit))

    total_tasks = len(tasks)
    logging.info(f"总共需要运行 {total_tasks} 个任务组合。")

    # 3. 使用 ProcessPoolExecutor 并行执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        if MAX_WORKERS:
            logging.info(f"使用 {MAX_WORKERS} 个工作进程开始并行执行...")
        else:
            try:
                logging.info(f"使用 {executor._max_workers} 个CPU核心开始并行执行...")
            except AttributeError:
                logging.info(f"使用所有可用的CPU核心开始并行执行...")

        list(executor.map(run_single_experiment, tasks))

    logging.info("--- 所有任务已完成 ---")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="CVRPLIB Experiment。"
    )

    # 添加 "method_name" 参数 (位置参数)
    parser.add_argument(
        "method_name",
        type=str,
        help="要运行的方法/可执行文件的名称 (例如: AILSII_CPU.jar, filo2, hgs-TV)"
    )

    # 添加 --start-idx 和 --end-idx 参数
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="要运行的实例文件列表的起始索引 (包含)。如果不指定，则从 0 开始。"
    )

    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="要运行的实例文件列表的结束索引 (不包含)。如果不指定，则运行到列表末尾。"
    )

    # --- [!] 新增参数: time-limit ---
    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_TIME_LIMIT,
        help=f"每个实验的最大运行时间限制 (单位: 秒)。默认值: {DEFAULT_TIME_LIMIT} 秒。"
    )
    # ----------------------------------------

    # 解析命令行参数
    args = parser.parse_args()

    # 使用从命令行获取的参数调用 main 函数
    main(
        method_name=args.method_name,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        time_limit=args.time_limit  # [!] 传递 time_limit
    )