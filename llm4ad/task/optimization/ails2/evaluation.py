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

__all__ = ['Ails2Evaluation']


class Ails2Evaluation(Evaluation):
    """Evaluator for AILSII Java."""

    def __init__(self, timeout_seconds=10, dump_java_output_on_finish: bool = False, **kwargs):
        """
            Args:
                - 'dimension' (int): The dimension of tested case (default is 15).
                - 'weight' (int): The wight of tested case (default is 10).
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )
        # 将 timeout_seconds 存为成员变量，以便 run_command 使用
        self.timeout_seconds = timeout_seconds
        self.dump_java_output_on_finish = dump_java_output_on_finish
        self.java_commands = []  # 用于存储生成的 Java 命令

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

    def run_command(self, commands) -> Tuple[str, int]:
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
            full_stdout, full_stderr = process.communicate(timeout=self.timeout_seconds)

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
            print(f"Java process {commands[0]}... timed out (> {self.timeout_seconds}s). Terminating...",
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

    # def run_command(self, commands) -> Tuple[str, int]:
    #     """
    #     使用 Popen 启动一个 Java 子进程，实时流式传输其输出，
    #     并捕获最后一行输出中的分数部分（假定格式为 '时间;分数'）。
    #     """
    #     last_line_score = ""  # 只存储最后的分数
    #     full_stdout = []  # 用于在出错时转储
    #     full_stderr = []  # 用于在出错时转储
    #
    #     try:
    #         # 1. 启动 Popen 进程
    #         process = subprocess.Popen(
    #             commands,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             text=True,
    #             encoding='utf-8',  # 明确指定编码
    #             bufsize=1  # 行缓冲
    #         )
    #
    #         # 2. 实时读取 stdout
    #         if process.stdout:
    #             for line in iter(process.stdout.readline, ''):
    #                 line_full = line.strip()
    #                 if not line_full:  # 跳过空行
    #                     continue
    #
    #                 if self.realtime_print_java_output:
    #                     # 实时打印到 Python 控制台
    #                     print(f"[Java]: {line_full}")
    #
    #                 # --- 分数解析逻辑 ---
    #                 # 捕获最后一行非空行，并只取分号后的部分
    #                 try:
    #                     # 假设格式为 "时间;分数"
    #                     score_part = line_full.split(';', 1)[1].strip()
    #                     if score_part:  # 确保分号后有内容
    #                         last_line_score = score_part
    #                 except IndexError:
    #                     # 如果某行没有分号, 保持上一行的 last_line_score 不变
    #                     # 忽略没有分号的行 (比如 "编译成功", "文件读取错误" 等)
    #                     pass
    #                     # --- 分数解析逻辑结束 ---
    #
    #                 full_stdout.append(line)
    #             process.stdout.close()  # 关闭流
    #
    #         # 3. 实时读取 stderr
    #         if process.stderr:
    #             for line in iter(process.stderr.readline, ''):
    #                 line_stripped = line.strip()
    #                 if self.realtime_print_java_output:
    #                     # 实时打印到 Python 控制台
    #                     print(f"[Java ERR]: {line_stripped}", file=sys.stderr)
    #                 full_stderr.append(line)
    #             process.stderr.close()  # 关闭流
    #
    #         # 4. 等待进程结束（并处理超时）
    #         return_code = process.wait(timeout=self.timeout_seconds)
    #
    #         # 5. 检查返回码和最后一行
    #         if return_code != 0:
    #             print(f"Java process {commands[0]}... failed. Return code: {return_code}", file=sys.stderr)
    #             if not self.realtime_print_java_output:
    #                 print("--- Java Stderr DUMP ---", file=sys.stderr)
    #                 print("".join(full_stderr), file=sys.stderr)
    #                 print("--------------------------", file=sys.stderr)
    #             return "CRASH", return_code
    #
    #         if not last_line_score:
    #             print(f"Java process {commands[0]}... succeeded but captured no valid score output.", file=sys.stderr)
    #             if not self.realtime_print_java_output:  # 如果实时打印关闭了，转储 stdout 帮助调试
    #                 print("--- Java Stdout DUMP ---", file=sys.stderr)
    #                 print("".join(full_stdout), file=sys.stderr)
    #                 print("--------------------------", file=sys.stderr)
    #             return "NO_SCORE_OUTPUT", return_code
    #
    #         # 成功
    #         return last_line_score, return_code
    #
    #     except subprocess.TimeoutExpired:
    #         print(f"Java process {commands[0]}... timed out (> {self.timeout_seconds}s). Terminating...", file=sys.stderr)
    #         process.kill()
    #         try:
    #             _stdout_data, _stderr_data = process.communicate(timeout=5)
    #             if self.realtime_print_java_output:
    #                 print(f"[Java TIMEOUT STDOUT]: {_stdout_data}")
    #                 print(f"[Java TIMEOUT STDERR]: {_stderr_data}", file=sys.stderr)
    #         except Exception as e:
    #             print(f"Error while terminating timed-out process: {e}", file=sys.stderr)
    #         return "TIMEOUT", -1
    #
    #     except Exception as e:
    #         print(f"Python error during run_command: {e}", file=sys.stderr)
    #         if 'process' in locals() and process.poll() is None:
    #             process.kill()
    #         return "PY_ERROR", -2

    def evaluate(self, java_script: str, subprocess_index=0) -> float | None:
        current_path = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_path, java_dir + f"_{subprocess_index}")  # Java 项目沙盒根目录
        # target_dir = os.path.join(current_path, java_dir)  # Java 项目沙盒根目录
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
            instances_dir = os.path.join(target_dir, "XLDemo_eohtest")
            instance_files = glob.glob(os.path.join(instances_dir, "*.vrp"))

            if not instance_files:
                print(f"Error: No .vrp instance files were found in {instances_dir}.", file=sys.stderr)
                return None

            main_class = "SearchMethod.AILSII"

            # 根据原始命令，为每个实例生成运行命令
            instance_commands = []
            for instance_file_path in instance_files:
                # 仿照原始命令: java -jar AILSII.jar -file data/X-n214-k11.vrp -rounded true -best 10856 -limit 100 -stoppingCriterion Time
                # 我们省略 -best 参数，因为它用于比较，而不是运行
                command = [
                    "java",
                    "-cp", classpath_run,
                    main_class,
                    "-file", instance_file_path,
                    "-rounded", "true",
                    "-limit", str(self.timeout_seconds),  # 使用 __init__ 中的超时设置
                    "-stoppingCriterion", "Time"
                ]
                instance_commands.append(command)

        # --- 2. 设置JDK环境 ---
        if "JDK_PATH" not in os.environ:
            jdk_bin_path = r";C:\Program Files\Common Files\Oracle\Java\javapath"  # <== 示例路径，请修改
            os.environ['PATH'] += jdk_bin_path
            os.environ["JDK_PATH"] = "set"

        # --- 3. 注入和编译 ---
        # 【!!!】假设 Java 源代码在 'Method/AILS-II/src' 目录下
        # 这基于你的代码骨架。如果不对请修改。
        src_path = os.path.join(target_dir, "Method", "AILS-II", "src")
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
            text=True
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

        # --- 4. 【修改】串行评估 ---
        try:
            print(f"Subprocess {subprocess_index}: Starting serial evaluation of {len(instance_commands)} instances...")
            results = []
            for cmd in instance_commands:
                result = self.run_command(cmd)
                results.append(result)

            # 打印结果
            fitness = []
            for (last_line, return_code) in results:
                fitness.append(last_line)

            # 转换为浮点数
            fitness_values = [float(e) for e in fitness]

            # 计算平均值
            final_fitness = np.mean(fitness_values)

            print(f"Subprocess {subprocess_index}: Evaluation complete. Average fitness: {final_fitness}")
            return float(final_fitness)

        except Exception as e:
            print(f"Serial evaluation failed: {e}", file=sys.stderr)
            return None

    def evaluate_program(self, program_str: str, **kwargs) -> Any | None:
        subprocess_index = kwargs.get("subprocess_index", 0)
        return self.evaluate(program_str, subprocess_index=subprocess_index)


if __name__ == '__main__':
    java_script = """
/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Perturbation;

import java.util.HashMap;
import java.util.Random;

import Data.Instance;
import DiversityControl.OmegaAdjustment;
import Improvement.IntraLocalSearch;
import SearchMethod.Config;
import Solution.Node;
import Solution.Route;
import Solution.Solution;

public abstract class Perturbation 
{
	protected Route routes[];
	protected int numRoutes;
	protected Node solution[];
	protected double f=0;
	protected Random rand=new Random();
	public double omega;
	OmegaAdjustment chosenOmega;
	Config config;
	protected Node candidates[];
	protected int countCandidates;

	InsertionHeuristic[]insertionHeuristics;
	public InsertionHeuristic selectedInsertionHeuristic;
	
	Node node;
	
	public PerturbationType perturbationType;
	int size;
	HashMap<String, OmegaAdjustment> omegaSetup;
	
	double bestCost,bestDist;
	int numIterUpdate;
	int indexHeuristic;
	
	double cost,dist;
	double costPrev;
	int indexA,indexB;
	Node bestNode,aux;
	Instance instance;
	int limitAdj;
	
	IntraLocalSearch intraLocalSearch;
	
	public Perturbation(Instance instance,Config config,
	HashMap<String, OmegaAdjustment> omegaSetup, IntraLocalSearch intraLocalSearch) 
	{
		this.config=config;
		this.instance=instance;
		this.insertionHeuristics=config.getInsertionHeuristics();
		this.size=instance.getSize()-1;
		this.candidates=new Node[size];
		this.omegaSetup=omegaSetup;
		this.numIterUpdate=config.getGamma();
		this.limitAdj=config.getVarphi();
		this.intraLocalSearch=intraLocalSearch;
	}
	
	public void setOrder()
	{
		Node aux;
		for (int i = 0; i < countCandidates; i++)
		{
			indexA=rand.nextInt(countCandidates);
			indexB=rand.nextInt(countCandidates);
			
			aux=candidates[indexA];
			candidates[indexA]=candidates[indexB];
			candidates[indexB]=aux;
		}
	}
	
	public void applyPerturbation(Solution s){}
	
	protected void setSolution(Solution s)
	{
		this.numRoutes=s.getNumRoutes();
		this.routes=s.routes;
		this.solution=s.getSolution();
		this.f=s.f;
		for (int i = 0; i < numRoutes; i++) 
		{
			routes[i].modified=false;
			routes[i].first.modified=false;
		}
		
		for (int i = 0; i < size; i++) 
			solution[i].modified=false;
	
		indexHeuristic=rand.nextInt(insertionHeuristics.length);
		selectedInsertionHeuristic=insertionHeuristics[indexHeuristic];
		
		chosenOmega=omegaSetup.get(perturbationType+"");
		omega=chosenOmega.getActualOmega();
		omega=Math.min(omega, size);
		
		countCandidates=0;
	}
	
	protected void assignSolution(Solution s)
	{
		s.f=f;
		s.numRoutes=this.numRoutes;
	}
	
	protected Node getNode(Node no)
	{
		switch(selectedInsertionHeuristic)
		{
			case Distance: return getBestKNNNo2(no,1);
			case Cost: return getBestKNNNo2(no,limitAdj);
		}
		return null;
	}
	
	protected Node getBestKNNNo2(Node no,int limit)
	{
		bestCost=Double.MAX_VALUE;
		boolean flag=false;
		bestNode=null;
		
		int count=0;
		flag=false;
		for (int i = 0; i < no.knn.length&&count<limit; i++) 
		{
			if(no.knn[i]==0)
			{
				for (int j = 0; j < numRoutes; j++) 
				{
					aux=routes[j].first;
					flag=true;
					cost=instance.dist(aux.name,no.name)+instance.dist(no.name,aux.next.name)-instance.dist(aux.name,aux.next.name);
					if(cost<bestCost)
					{
						bestCost=cost;
						bestNode=aux;
					}
				}
				if(flag)
					count++;
			}
			else
			{
				aux=solution[no.knn[i]-1];
				if(aux.nodeBelong)
				{
					count++;
					cost=instance.dist(aux.name,no.name)+instance.dist(no.name,aux.next.name)-instance.dist(aux.name,aux.next.name);
					if(cost<bestCost)
					{
						bestCost=cost;
						bestNode=aux;
					}
				}
			}
		}
		
		if(bestNode==null)
		{
			for (int i = 0; i < solution.length; i++) 
			{
				aux=solution[i];
				if(aux.nodeBelong)
				{
					cost=instance.dist(aux.name,no.name)+instance.dist(no.name,aux.next.name)-instance.dist(aux.name,aux.next.name);
					if(cost<bestCost)
					{
						bestCost=cost;
						bestNode=aux;
					}
				}
			}
		}
		
		if(bestNode==null)
		{
			for (int i = 0; i < solution.length; i++) 
			{
				aux=solution[i];
				if(aux.nodeBelong)
				{
					cost=instance.dist(aux.name,no.name)+instance.dist(no.name,aux.next.name)-instance.dist(aux.name,aux.next.name);
					if(cost<bestCost)
					{
						bestCost=cost;
						bestNode=aux;
					}
				}
			}
		}
		
		cost=instance.dist(bestNode.name,no.name)+instance.dist(no.name,bestNode.next.name)-instance.dist(bestNode.name,bestNode.next.name);
		costPrev=instance.dist(bestNode.prev.name,no.name)+instance.dist(no.name,bestNode.name)-instance.dist(bestNode.prev.name,bestNode.name);
		if(cost<costPrev)
		{
			return bestNode;
		}
		else
		{
			return bestNode.prev;
		}
	}
	
	public void addCandidates() 
	{
		for (int i = 0; i < countCandidates; i++) 
		{
			node=candidates[i];
			bestNode=getNode(node);
			
			f+=bestNode.route.addAfter(node, bestNode);
		}
	}
	
	public int getIndexHeuristic() {
		return indexHeuristic;
	}

	public OmegaAdjustment getChosenOmega() {
		return chosenOmega;
	}
	
	public PerturbationType getPerturbationType() {
		return perturbationType;
	}
	
}
    """

    eval = Ails2Evaluation(timeout_seconds=5, dump_java_output_on_finish=True)  # 设置超时

    print("开始评估...")
    res = eval.evaluate_program(java_script, None)
    print(f"评估完成。平均适应度: {res}")
