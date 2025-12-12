from __future__ import annotations

from typing import Any, List, Tuple
import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.ails2.template import template_program, task_description, aim_java_relative_path, \
    java_dir  # java_dir = CVRPLIB-2025-AILSII 是为了多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
import os  # aim_java_relative_path 是被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"
import subprocess
import sys
import glob
import shutil
import re
import concurrent.futures # 导入 concurrent.futures

__all__ = ['Ails2Evaluation']


class Ails2Evaluation(Evaluation):
    """Evaluator for AILSII Java."""

    def __init__(self, default_instance_timeout=10,
                 dump_java_output_on_finish: bool = False,
                 jdk_bin_path: str = r";C:\Program Files\Common Files\Oracle\Java\javapath",            # TODO 1
                 max_parallel_instances: int = 4,
                 instances_subdir: str = "XLDemo_eohtest",  # AILSII 实例子目录
                 test_mode: bool = False,       # 如果只是看跑不跑得通，设为True，那每次评估只有15s.
                 **kwargs):
        """
            Args:
                - 'dimension' (int): The dimension of tested case (default is 15).
                - 'weight' (int): The wight of tested case (default is 10).
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=default_instance_timeout
        )

        self.default_instance_timeout = default_instance_timeout
        self.dump_java_output_on_finish = dump_java_output_on_finish
        self.java_commands = []  # 将存储 (command, python_timeout) 元组
        self.jdk_bin_path = jdk_bin_path
        self.max_parallel_instances = max(1, max_parallel_instances)
        self.instances_subdir = instances_subdir  # 存储实例子目录
        self.test_mode = test_mode

        # 3. 【核心修正】预计算总时间并覆盖 self.timeout_seconds
        try:
            # 【修改】现在返回 (total, max) 两个值
            total_serial_time, max_instance_time = self._calculate_total_time(
                self.instances_subdir,
                self.default_instance_timeout
            )

            if total_serial_time == 0:
                print(f"Warning: Could not find/parse any .vrp files in source directory. Using default timeout.",
                      file=sys.stderr)
                estimated_robust_time = self.default_instance_timeout
            else:
                # 【修正公式】 (总时间 / 并行数) + 最长时间 (防止“落后者”问题)
                # 这是一个更保守、更安全的上限估算
                estimated_robust_time = (total_serial_time / self.max_parallel_instances) + max_instance_time

            # 增加 60 秒的编译缓冲 和 10% 的调度/方差缓冲
            compilation_buffer = 60
            variance_buffer_percent = 0.10

            final_framework_timeout = (estimated_robust_time * (1 + variance_buffer_percent)) + compilation_buffer

            # 【关键】覆盖父类的 timeout_seconds 属性
            # SecureEvaluator 将读取这个新值
            self.timeout_seconds = final_framework_timeout

            print(f"Ails2Evaluation Initialized (Robust Timeout):")
            print(f"  - Total serial time calculated: {total_serial_time:.2f} s")
            print(f"  - Max instance time: {max_instance_time:.2f} s")
            print(f"  - L2 Parallelism: {self.max_parallel_instances} workers")
            print(f"  - Estimated robust parallel time: {estimated_robust_time:.2f} s")
            print(f"  - ==> Setting SecureEvaluator framework timeout to: {self.timeout_seconds:.2f} s")

        except Exception as e:
            print(f"FATAL ERROR during Ails2Evaluation __init__ time calculation: {e}", file=sys.stderr)
            # 出现异常时，回退到一个非常大的硬编码值（3小时），以防万一
            self.timeout_seconds = 10800

    def _calculate_total_time(self, instances_subdir: str, default_timeout: int) -> Tuple[float, float]:
        """
        扫描 *源* VRP 目录, 累加所有实例的动态时间。
        【修改】: 同时返回 total_time 和 max_time。
        """
        total_time = 0.0
        max_time = 0.0

        try:
            current_path = os.path.dirname(os.path.abspath(__file__))
            # 查找 *源* 目录, 而不是_0, _1...
            src_java_dir = os.path.join(current_path, java_dir)
            src_instances_dir = os.path.join(src_java_dir, instances_subdir)

            if not os.path.isdir(src_instances_dir):
                print(f"Warning: Source instance directory not found at: {src_instances_dir}", file=sys.stderr)
                return default_timeout, default_timeout

            instance_files = glob.glob(os.path.join(src_instances_dir, "*.vrp"))

            if not instance_files:
                print(f"Warning: No .vrp files found in {src_instances_dir}", file=sys.stderr)
                return default_timeout, default_timeout

            all_times = []
            for instance_file_path in instance_files:
                filename = os.path.basename(instance_file_path)
                match = re.search(r'n(\d+)', filename)

                java_limit_time = default_timeout  # 默认

                if match:
                    try:
                        node_count = int(match.group(1))
                        if self.test_mode:
                            calculated_time = 15
                        else:
                            calculated_time = (node_count // 25) * 60


                        if calculated_time > java_limit_time:  # 只有在计算时间 > 默认时才覆盖
                            java_limit_time = calculated_time

                    except (ValueError, IndexError):
                        pass  # 保持 default_timeout

                # Python 的硬超时 = Java的限制 + 5秒缓冲
                python_timeout = java_limit_time + 120
                all_times.append(python_timeout)

            if not all_times:
                return default_timeout, default_timeout

            total_time = sum(all_times)
            max_time = max(all_times)

            return total_time, max_time

        except Exception as e:
            print(f"Error in _calculate_total_time: {e}", file=sys.stderr)
            return default_timeout, default_timeout  # 出错时返回默认值


    def copy_dir_multiple_times(self, n: int):
        """
        Copy `src_dir` N times into `dst_base_dir`,
        appending _i to each new directory name.

        Args:
            src_dir: str, path to the source directory
            dst_base_dir: str, path where copies will be created
            n: int, number of copies
        """

        dst_base_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(dst_base_dir, java_dir)

        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"Source directory {src_dir} does not exist.")

        os.makedirs(dst_base_dir, exist_ok=True)

        for i in range(n):
            # 构造目标目录名
            dst_dir = os.path.join(dst_base_dir, os.path.basename(src_dir) + f"_{i}")
            # 如果目标目录已存在，可以选择覆盖或跳过
            if os.path.exists(dst_dir):
                print(f"{dst_dir} already exists, removing it first.")
                shutil.rmtree(dst_dir)
            # 复制整个目录
            shutil.copytree(src_dir, dst_dir)
            print(f"Copied {src_dir} -> {dst_dir}")

    def run_command(self, commands, python_timeout: int) -> Tuple[str, int]:
        """
        [重构] 使用 Popen.communicate() 来正确施加超时，
        并能在超时或正常结束时解析最后的分数。
        """
        last_line_score = ""
        full_stdout = ""
        full_stderr = ""
        process = None  # 确保 process 在 except 块中可访问

        try:
            # 1. 启动 Popen 进程
            process = subprocess.Popen(
                commands,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )

            # 2. 使用 communicate() 来处理 stdout/stderr 并施加超时
            # 这会等待进程结束，或在超时后抛出 TimeoutExpired
            # 它会一次性读取所有输出
            full_stdout, full_stderr = process.communicate(timeout=python_timeout)

            return_code = process.returncode

            # 3. 检查返回码 (正常退出)
            if return_code != 0:
                print(f"Java process {commands[0]}... failed. Return code: {return_code}", file=sys.stderr)
                if not self.dump_java_output_on_finish:
                    print("--- Java Stderr DUMP ---", file=sys.stderr)
                    print(full_stderr, file=sys.stderr)
                    print("--------------------------", file=sys.stderr)
                return "CRASH", return_code

        except subprocess.TimeoutExpired as e:
            print(f"Java process {commands[0]}... timed out (> {python_timeout}s). Terminating...",
                  file=sys.stderr)
            process.kill()  # 确保进程被终止

            # communicate() 在超时后，可能已经捕获了部分输出
            # 我们再次调用它（设置短超时）来获取所有残留的缓冲输出
            try:
                full_stdout, full_stderr = process.communicate(timeout=5)
            except Exception as e_inner:
                print(f"Error during post-timeout communicate: {e_inner}", file=sys.stderr)
                # 即使这里失败，full_stdout 可能在 e.stdout 中，但更安全的是依赖第一次调用
                # 为简单起见，我们假设第二次调用能取到（或者第一次超时时已填充）
                pass  # 即使失败，也继续尝试解析 full_stdout

            return_code = -1  # 标记为超时

            # ★ 您的核心需求：即使超时，也解析输出
            # fall-through 到下面的解析逻辑

        except Exception as e:
            print(f"Python error during run_command: {e}", file=sys.stderr)
            if process and process.poll() is None:
                process.kill()
            return "PY_ERROR", -2

        # --- 通用解析逻辑 (无论正常结束还是超时) ---

        if self.dump_java_output_on_finish:
            print(f"[Java STDOUT DUMP]:\n{full_stdout}")
            if full_stderr:
                print(f"[Java STDERR DUMP]:\n{full_stderr}", file=sys.stderr)

        if full_stdout:
            # 从后向前查找最后一个有效的分数行
            for line in reversed(full_stdout.strip().split('\n')):
                line_full = line.strip()
                if not line_full:
                    continue
                try:
                    score_part = line_full.split(';', 1)[1].strip()
                    if score_part:
                        last_line_score = score_part
                        break  # 找到最后一个，跳出
                except IndexError:
                    pass  # 忽略无效行 (如 "编译成功" 等)

        # 4. 根据解析结果返回
        if last_line_score:
            if return_code == -1:  # 如果是超时
                print(f"Timeout: Captured last score before kill: {last_line_score}")
            # ★ 关键：返回最后的分数，而不是 "TIMEOUT" 字符串
            return last_line_score, return_code
        else:
            # 运行成功/超时，但没有捕获到任何分数
            if return_code == -1:
                return "TIMEOUT_NO_SCORE", return_code
            else:
                return "NO_SCORE_OUTPUT", return_code

    def evaluate(self, java_script: str, subprocess_index=0) -> float | None:
        current_path = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_path, java_dir + f"_{subprocess_index}")  # Java 项目沙盒根目录
        target_change_java = os.path.join(target_dir, aim_java_relative_path)  # 被 LLM 修改的 Java 文件

        # 自动处理 Windows (;) 和 Linux (:) 的 classpath 分隔符
        classpath_separator = ';' if sys.platform == "win32" else ':'

        # 编译输出目录 (存放 .class 文件)
        compile_output_dir = os.path.join(target_dir, "bin")
        os.makedirs(compile_output_dir, exist_ok=True)  # 确保 bin 目录存在

        libs_dir = os.path.join(target_dir, "libs")
        libs_path_glob = os.path.join(libs_dir, "*")

        if os.path.isdir(libs_dir) and glob.glob(libs_path_glob):
            print(f"Subprocess {subprocess_index}: 'libs' directory found. Adding dependencies.")
            classpath_compile = libs_path_glob
            classpath_run = f"{compile_output_dir}{classpath_separator}{libs_path_glob}"
        else:
            print(
                f"Subprocess {subprocess_index}: 'libs' directory not found in {target_dir}. Assuming no dependencies.")
            classpath_compile = ""
            classpath_run = f"{compile_output_dir}"  # 运行时只包含 bin 目录

        # --- 1. 构建Java命令 ---
        if not self.java_commands:
            print(f"Subprocess {subprocess_index}: First run, building Java command list...")
            instances_dir = os.path.join(target_dir, self.instances_subdir)                                # TODO 2 此处是AILSII需要评估的实例存放目录
            instance_files = glob.glob(os.path.join(instances_dir, "*.vrp"))

            if not instance_files:
                print(f"Error: No .vrp instance files were found in {instances_dir}.", file=sys.stderr)
                return None

            main_class = "SearchMethod.AILSII"                                                        # TODO 3 如果还是AILSII就不需要改

            # 根据原始命令，为每个实例生成运行命令
            instance_commands = []
            for instance_file_path in instance_files:                                                 # TODO 4  如果还是AILSII就不需要改
                # --- 3. START: 动态时间计算 ---
                filename = os.path.basename(instance_file_path)
                match = re.search(r'n(\d+)', filename)

                java_limit_time = self.default_instance_timeout

                if match:
                    try:
                        node_count = int(match.group(1))
                        if self.test_mode:
                            calculated_time = 15
                        else:
                            calculated_time = (node_count // 25) * 60

                        if calculated_time > java_limit_time:  # 只有在计算时间 > 默认时才覆盖
                            java_limit_time = calculated_time

                    except (ValueError, IndexError):
                        pass  # 使用 default

                python_timeout = java_limit_time + 120

                # 仿照原始命令创建command, 比如: java -jar AILSII.jar -file data/X-n214-k11.vrp -rounded true -best 10856 -limit 100 -stoppingCriterion Time
                command = [
                    "java",
                    "-cp", classpath_run,
                    main_class,
                    "-file", instance_file_path,
                    "-rounded", "true",
                    "-limit", str(java_limit_time),  # 使用 __init__ 中的超时设置
                    "-stoppingCriterion", "Time"
                ]
                instance_commands.append( (command, python_timeout) )

            self.java_commands = instance_commands
            print(f"Java command list built with dynamic timeouts.")

        # --- 2. 设置JDK环境 ---
        if "JDK_PATH" not in os.environ:
            jdk_bin_path = self.jdk_bin_path  # <== 示例路径，请修改
            os.environ['PATH'] += jdk_bin_path
            os.environ["JDK_PATH"] = "set"

        # --- 3. 注入和编译 ---
        # 【!!!】假设 Java 源代码在 'Method/AILS-II/src' 目录下
        # 这基于你的代码骨架。如果不对请修改。

        # src_path = os.path.join(target_dir, "Method", "AILS-II", "src")
        src_path = os.path.join(target_dir, "src", "AILS-II_origin", "src")    # TODO 5  如果还是AILSII就不需要改
        sources_file = os.path.join(target_dir, "sources.txt")

        # 注入 LLM 生成的 Java 代码
        try:
            with open(target_change_java, "w", encoding='utf-8') as f:
                f.write(java_script)
        except Exception as e:
            print(f"Failed to write Java file: {e}", file=sys.stderr)
            return None

        # 【改进】使用 glob 收集所有 .java 文件，实现跨平台
        try:
            java_files = glob.glob(os.path.join(src_path, "**/*.java"), recursive=True)
            if not java_files:
                print(f"Error: No .java source files found in {src_path}.", file=sys.stderr)
                return None

            with open(sources_file, 'w', encoding='utf-8') as fs:
                fs.write("\n".join(java_files))
        except Exception as e:
            print(f"Failed to collect Java source files: {e}", file=sys.stderr)
            return None

        # 编译命令
        compile_cmd = [
            "javac",
            "-d", compile_output_dir,
            "-sourcepath", src_path,
            # f"@{sources_file}"
        ]
        if classpath_compile:
            compile_cmd.extend(["-cp", classpath_compile])
        compile_cmd.append(f"@{sources_file}")  # 把 @sources_file 放到最后

        if self.dump_java_output_on_finish:
            print(f"Subprocess {subprocess_index}: Executing compile command: {' '.join(compile_cmd)}")

        # 运行编译 (这是一个单独的 subprocess 调用，是允许的)
        compile_process = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',  # 显式指定编码
            errors='replace'  # <--- 【重点】防止编译错误信息里的中文导致崩溃
        )

        try:
            if compile_process.returncode != 0:
                print(f"Subprocess {subprocess_index}: Compilation failed! Return code: {compile_process.returncode}",
                      file=sys.stderr)
                print("--- Javac Stderr DUMP ---", file=sys.stderr)
                print(compile_process.stderr, file=sys.stderr)
                print("---------------------------", file=sys.stderr)
                return None  # 编译失败，返回极差分数
            else:
                # print("Compilation succeeded!") # 成功，安静处理
                pass
        except Exception as e:
            print(f"Error during compilation check: {e}", file=sys.stderr)
            return None

        # --- 4. 【修改】并行评估 ---
        try:
            print(
                f"Subprocess {subprocess_index}: Starting PARALLEL evaluation of {len(self.java_commands)} instances (max_workers={self.max_parallel_instances})...")
            results = []
            futures = []

            # 使用 ThreadPoolExecutor 来管理并发的 subprocesses
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_instances) as executor:
                # 提交所有任务
                for cmd, cmd_timeout in self.java_commands:
                    if self.dump_java_output_on_finish:
                        instance_name = os.path.basename(cmd[6])
                        java_limit = cmd[10]
                        print(
                            f"Submitting: {instance_name} (Java Limit: {java_limit}s, Python Timeout: {cmd_timeout}s)")

                    # 提交任务到线程池 (self.run_command 是线程安全的)
                    futures.append(executor.submit(self.run_command, cmd, cmd_timeout))

                # 按完成顺序收集结果
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()  # (last_line, return_code)
                        results.append(result)
                    except Exception as e:
                        print(f"Error collecting future result: {e}", file=sys.stderr)
                        results.append(("FUTURE_ERROR", -99))

            # --- 结果聚合 ---
            if not results:
                print(f"Subprocess {subprocess_index}: No results collected from parallel evaluation.",
                      file=sys.stderr)
                return None

            # 打印结果
            fitness = []
            valid_results_count = 0
            for (last_line, return_code) in results:
                try:
                    fitness_value = float(last_line)
                    fitness.append(fitness_value)
                    valid_results_count += 1
                except (ValueError, TypeError):
                    print(f"Warning: Discarding invalid result '{last_line}' (Code: {return_code})", file=sys.stderr)

            if not fitness:
                print(f"Subprocess {subprocess_index}: Evaluation failed. No valid fitness scores found.",
                      file=sys.stderr)
                return None  # 所有实例都失败了

            expected_count = len(self.java_commands)
            actual_count = len(fitness)
            # 3. 【核心修改】如果不一致，直接丢弃本次评估
            if actual_count != expected_count:
                print(f"Subprocess {subprocess_index}: Stability Check Failed!", file=sys.stderr)
                print(f"   - Expected: {expected_count} instances", file=sys.stderr)
                print(f"   - Success : {actual_count} instances", file=sys.stderr)
                print(f"   - Action  : Discarding evaluation (returning None).", file=sys.stderr)
                return None

            final_fitness = np.mean(fitness)

            print(f"Subprocess {subprocess_index}: Evaluation complete. Average fitness: {final_fitness}")
            return -float(final_fitness)

        except Exception as e:
            print(f"Serial evaluation failed: {e}", file=sys.stderr)
            return None

    def evaluate_program(self, program_str: str, **kwargs) -> Any | None:
        subprocess_index = kwargs.get("subprocess_index", 0)
        return self.evaluate(program_str, subprocess_index=subprocess_index)


if __name__ == '__main__':
    java_script = """
package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment 
{
	int distMMin;
	int distMMax;
	int iterator;
	long ini;
	double executionMaximumLimit;
	double alpha=1;
	StoppingCriterionType stoppingCriterionType;
	IdealDist idealDist;

	public DistAdjustment(IdealDist idealDist,Config config,double executionMaximumLimit) 
	{
		this.idealDist=idealDist;
		this.executionMaximumLimit=executionMaximumLimit;
		this.distMMin=config.getDMin();
		this.distMMax=config.getDMax();
		this.idealDist.idealDist=distMMax;
		this.stoppingCriterionType=config.getStoppingCriterionType();
	}

	public void distAdjustment()
	{
		if(iterator==0)
			ini=System.currentTimeMillis();
		
		iterator++;
		
		switch(stoppingCriterionType)
		{
			case Iteration: 	iterationAdjustment(); break;
			case Time: timeAdjustment(); break;
			default:
				break;
								
		}
		
		idealDist.idealDist*=alpha;
		idealDist.idealDist= Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
		
	}
	
	private void iterationAdjustment()
	{
		alpha=Math.pow((double)distMMin/(double)distMMax, (double) 1/executionMaximumLimit);
	}
	
	private void timeAdjustment()
	{
		double current=(double)(System.currentTimeMillis()-ini)/1000;
		double timePercentage=current/executionMaximumLimit;
		double total=(double)iterator/timePercentage;
		alpha=Math.pow((double)distMMin/(double)distMMax, (double) 1/total);
	}
}
    """



    eval = Ails2Evaluation(dump_java_output_on_finish=True)  # 设置超时

    # ================= 核心修改 START =================
    print("正在初始化沙盒环境 (复制项目文件夹)...")
    try:
        # 这里的参数 1 表示复制 1 份，生成名为 "..._0" 的目录
        # 这正好对应 evaluate_program 默认的 subprocess_index=0
        eval.copy_dir_multiple_times(1)
    except Exception as e:
        print(f"环境初始化失败: {e}")
        # 如果源文件夹都不存在，后面肯定跑不通，建议直接退出
        sys.exit(1)
    print("沙盒环境初始化完成。")
    # ================= 核心修改 END ===================

    print("开始评估...")
    res = eval.evaluate_program(java_script)
    print(f"评估完成。平均适应度: {res}")
