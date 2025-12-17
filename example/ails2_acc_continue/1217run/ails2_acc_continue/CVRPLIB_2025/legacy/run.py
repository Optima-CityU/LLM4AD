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

TIME_LIMIT = 5 * 24 * 3600  # 5天，单位为秒
# TIME_LIMIT = 3

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


def get_command_args(method_path: str, instance_path: str) -> Optional[List[str]]:
    """
    [!] 这是您需要“手工指定”参数的核心函数 (要求 1)

    根据传入的 method_path，返回完整的命令行参数列表。

    参数:
        method_path (str): 可执行文件的完整路径 (例如, "/bin/methodA")
        instance_path (str): 实例文件的完整路径 (例如, "/XLTEST/instance01.vrp")

    返回:
        List[str]: 一个字符串列表，例如 ["/bin/methodA", "-i", "/XLTEST/instance01.vrp", "--param", "value"]
        Optional[None]: 如果此方法未定义命令，返回None，将跳过执行
    """

    instance_name = os.path.basename(instance_path)

    # 获取可执行文件的基本名称，用于判断
    method_name = os.path.basename(method_path)

    # --- [!] 请您在此处配置 ---
    #
    # 请为您拥有的4个（或更多）method配置命令行
    #
    # 示例：

    if 'AILSII' in method_name:
        # 假设 methodA 的命令是: /bin/methodA -i <instance_file> --heuristic "GA"
        return ["java", "-jar", "-Xms2000m", "-Xmx4000m", f"bin/{method_name}", "-file", instance_path, "-rounded", "true", "-stoppingCriterion", "Time", "-limit", f"{TIME_LIMIT}"]

    elif method_name == "filo2":
        # 假设 methodB 的命令是: /bin/methodB --input <instance_file> --config /etc/config.xml -t 300
        return [f"{method_path}", instance_path, "-t", f"{TIME_LIMIT}", "--outpath", f"remote_results/{method_name}"]

    elif method_name == "hgs":
        # 假设 methodC 的命令是: /bin/methodC <instance_file>
        return [f"{method_path}", instance_path, f"remote_results/{method_name}/{instance_name}.sol", "-seed", "1", "-t", f"{TIME_LIMIT}"]

    elif method_name == "hgs-TV":
        return [f"{method_path}", instance_path, "-sol", f"remote_results/{method_name}/{instance_name}.sol", "-seed", "1", "-t", f"{TIME_LIMIT}", "-type", "Uchoa", "-bss", f"remote_results/{method_name}/{instance_name}.csv", "-deco", "BarycentreClustering"]

    # elif method_name == "methodD":
    #     # 假设 methodD 不接受参数，只运行
    #     # (虽然这不符合题意，但作为示例)
    #     # return [method_path]
    #
    #     # 假设 methodD 尚未配置，我们可以返回 None 来跳过它
    #     logging.warning(f"方法 {method_name} 的命令未在 get_command_args 中定义，已跳过。")
    #     return None

    else:
        # 其他未知的可执行文件
        logging.error(f"未知的可执行文件: {method_name}")
        return None

    # --- 配置结束 ---


def run_single_experiment(task_info: tuple) -> None:
    """
    运行单个实验（一个method + 一个instance）。
    此函数被设计为在单独的进程中运行，并包含完整的异常处理（要求 3）。

    [!] 特殊处理: AILSII_CPU.jar 的输出将被重定向到文件。
    """
    method_path, instance_path = task_info
    method_name = os.path.basename(method_path)
    instance_name = os.path.basename(instance_path)

    log_prefix = f"[{method_name} | {instance_name}]"

    try:
        # 1. 获取命令列表 (不含重定向)
        command_list = get_command_args(method_path, instance_path)

        if command_list is None:
            return

        logging.info(f"{log_prefix} 开始运行。命令: {' '.join(command_list)}")

        # --- [!] 核心修改点 ---

        if method_name == "AILSII_CPU.jar":
            #
            # --- 分支 1: AILSII_CPU.jar (重定向到 .csv 文件) ---
            #

            # 1a. 定义并创建输出目录
            output_dir = os.path.join(SCRIPT_DIR, "remote_results", method_name)
            output_csv_path = os.path.join(output_dir, f"{instance_name}.csv")
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                logging.error(f"{log_prefix} 无法创建输出目录 {output_dir}: {e}")
                return

            logging.info(f"{log_prefix} 输出将重定向到: {output_csv_path}")

            # 1b. 打开文件并执行
            with open(output_csv_path, 'w') as output_file:
                result = subprocess.run(
                    command_list,
                    stdout=output_file,  # [!] 将 stdout 导入文件
                    stderr=subprocess.PIPE,  # [!] 仍然捕获 stderr
                    text=True,
                    check=True,
                    # timeout=PROCESS_TIMEOUT
                )

            # 1c. 成功记录 (stdout 已在文件中)
            logging.info(f"{log_prefix} 运行成功。结果已保存到 {output_csv_path}")

        else:
            #
            # --- 分支 2: 所有其他方法 (捕获到内存) ---
            #

            result = subprocess.run(
                command_list,
                capture_output=True,  # [!] 捕获 stdout 和 stderr
                text=True,
                check=True,
                # timeout=PROCESS_TIMEOUT
            )

            # 2a. 成功记录 (stdout 在内存中)
            stdout_snippet = result.stdout.strip()[:200].replace('\n', ' ')
            logging.info(f"{log_prefix} 运行成功。STDOUT(snippet): {stdout_snippet}...")

        # --- 异常处理 (对两个分支都有效) ---

    except subprocess.CalledProcessError as e:
        logging.error(f"{log_prefix} 运行失败。返回码: {e.returncode}")
        # 无论哪个分支，e.stderr 都会被正确捕获 (如果是 PIPE) 或在 e.stderr (如果是 capture_output=True)
        if e.stderr:
            logging.error(f"{log_prefix} STDERR: {e.stderr.strip()}")

    # except subprocess.TimeoutExpired as e:
    #     logging.error(f"{log_prefix} 运行超时 (超过 {PROCESS_TIMEOUT} 秒)。进程已被终止。")
    #     if e.stderr:
    #         logging.error(f"{log_prefix} 超时前的 STDERR: {e.stderr.strip()}")

    except FileNotFoundError:
        logging.error(f"{log_prefix} 运行失败。命令或文件未找到: {command_list[0]}")

    except Exception as e:
        logging.error(f"{log_prefix} 发生意外的脚本错误: {e}")


def main(method_name: str = None):
    """主函数，负责编排所有任务"""
    METHOD_TO_RUN = os.path.join(METHODS_DIR, method_name) if method_name else None
    METHODS_OUTPUT_DIRS = os.path.join(OUTPUT_DIR, method_name)
    if not os.path.exists(METHODS_OUTPUT_DIRS):
        os.makedirs(METHODS_OUTPUT_DIRS, exist_ok=True)

    setup_logging()
    logging.info("--- 脚本开始运行 ---")

    # 1. 查找所有实例文件
    instance_pattern = os.path.join(INSTANCES_DIR, "*.vrp")
    instance_files = glob.glob(instance_pattern)

    if not METHOD_TO_RUN:
        logging.critical("配置错误：METHOD_TO_RUN 为空。请检查脚本配置。")
        return

    if not instance_files:
        logging.critical(f"配置错误：在 {INSTANCES_DIR} 中未找到任何 *.vrp 文件。")
        return

    logging.info(f"找到 {len(METHOD_TO_RUN)} 个可执行文件。")
    logging.info(f"找到 {len(instance_files)} 个实例文件。")

    # 2. 创建所有任务组合 (笛卡尔积)
    tasks = []
    for method in list([METHOD_TO_RUN]):
        for instance in instance_files:
            tasks.append((method, instance))

    total_tasks = len(tasks)
    logging.info(f"总共需要运行 {total_tasks} 个任务组合。")

    # 3. 使用 ProcessPoolExecutor 并行执行 (要求 2)
    # 它会自动管理进程池，并行运行 run_single_experiment 函数
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        if MAX_WORKERS:
            logging.info(f"使用 {MAX_WORKERS} 个工作进程开始并行执行...")
        else:
            # executor._max_workers 在 Python 3.8+ 可用
            try:
                logging.info(f"使用 {executor._max_workers} 个CPU核心开始并行执行...")
            except AttributeError:
                logging.info(f"使用所有可用的CPU核心开始并行执行...")

        # 使用 executor.map 来分发任务
        # map 会保持任务的原始顺序
        # 我们使用 list() 来强制执行所有任务并等待它们完成
        list(executor.map(run_single_experiment, tasks))

    logging.info("--- 所有任务已完成 ---")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="CVRPLIB Experiment。"
    )

    # 添加 "method_name" 参数
    # 这是一个 "位置参数" (positional argument)，因为没有 "--" 前缀
    # 它自动成为必需的 (required)
    parser.add_argument(
        "method_name",
        type=str,
        help="要运行的方法/可执行文件的名称 (例如: AILSII_CPU.jar, filo2, hgs-TV)"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 使用从命令行获取的 method_name 调用 main 函数
    # args.method_name 的值就是用户在命令行输入的值
    main(method_name=args.method_name)