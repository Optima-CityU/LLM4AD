from __future__ import annotations

import itertools
from typing import Any, List, Tuple
import numpy as np
# from fontTools.cffLib.specializer import commandsToProgram # 似乎未使用，注释掉

from llm4ad.base import Evaluation
from llm4ad.task.optimization.ails2.template import template_program, task_description, aim_java_relative_path, java_dir        # java_dir = CVRPLIB-2025-AILSII 是为了多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
import os                                                                                                                       # aim_java_relative_path 是被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"
import subprocess
import sys
import glob # 【新增】用于查找文件

__all__ = ['Ails2Evaluation']

class Ails2Evaluation(Evaluation):
    """Evaluator for online bin packing problem."""

    def __init__(self, timeout_seconds=180, dimension=15, weight=10, **kwargs):
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

    def run_command(self, commands):
        """
        【重写】执行单个java命令，增加了健壮的错误处理。
        """
        try:
            process = subprocess.run(
                commands,
                capture_output=True,
                text=True,
                # Java 程序的超时时间为 self.timeout_seconds
                # 我们给 Python 子进程一个稍大的超时缓冲
                timeout=self.timeout_seconds + 10
            )

            # 检查 Java 进程是否崩溃
            if process.returncode != 0:
                print(f"Java process failed with code {process.returncode}. STDERR: {process.stderr[:200]}...",
                      file=sys.stderr)
                return None, process.returncode  # 返回一个极差的适应度

            # 检查是否有标准输出
            if not process.stdout:
                print(f"Java process produced no output.", file=sys.stderr)
                return "999999", process.returncode  # 没有输出，返回坏分数

            # 假设最后一行是分数
            last_line = process.stdout.strip().splitlines()[-1]

            # 健壮性检查：确保最后一行是数字
            try:
                # 尝试解析分数，看它是否是一个合法的浮点数
                float(last_line)
                return last_line, process.returncode
            except (ValueError, IndexError):
                print(f"Java output was not a number: {last_line}", file=sys.stderr)
                return "999999", process.returncode  # 不是数字，返回坏分数

        except subprocess.TimeoutExpired:
            print(f"Java process timed out after {self.timeout_seconds + 10} seconds.", file=sys.stderr)
            return "999999", -1  # 超时，返回坏分数
        except Exception as e:
            print(f"run_command error: {e}", file=sys.stderr)
            return "999999", -2  # 其他Python错误，返回坏分数

    def evaluate(self, java_script: str, subprocess_index=0) -> float | None:
        # ... (步骤 1, 2, 3 保持不变) ...
        current_path = os.path.dirname(os.path.abspath(__file__))
        # target_dir = os.path.join(current_path, java_dir + f"_{subprocess_index}")  # Java 项目沙盒根目录
        target_dir = os.path.join(current_path, java_dir)  # Java 项目沙盒根目录
        target_change_java = os.path.join(target_dir, aim_java_relative_path)  # 被 LLM 修改的 Java 文件

        # --- 1. 构建Java命令 ---
        instances_dir = os.path.join(target_dir, "XLDemo")
        instance_files = glob.glob(os.path.join(instances_dir, "*.vrp"))

        if not instance_files:
            print(f"Error: No .vrp instance files were found in {instances_dir}.", file=sys.stderr)
            return None

        # 自动处理 Windows (;) 和 Linux (:) 的 classpath 分隔符
        classpath_separator = ';' if sys.platform == "win32" else ':'

        # 编译输出目录 (存放 .class 文件)
        compile_output_dir = os.path.join(target_dir, "bin")
        os.makedirs(compile_output_dir, exist_ok=True)  # 确保 bin 目录存在

        # 【!!!】假设 Java 依赖库在 'libs' 目录下
        # 请根据你的实际情况修改 'libs'
        libs_dir = os.path.join(target_dir, "libs")
        libs_path_glob = os.path.join(libs_dir, "*")

        # 动态检查 'libs' 目录是否存在且非空
        if os.path.isdir(libs_dir) and glob.glob(libs_path_glob):
            print(f"子进程 {subprocess_index}: 发现 'libs' 目录，添加依赖项。")
            classpath_compile = libs_path_glob
            classpath_run = f"{compile_output_dir}{classpath_separator}{libs_path_glob}"
        else:
            # 没有 libs 目录，设置为空
            print(f"子进程 {subprocess_index}: 未在 {target_dir} 中找到 'libs' 目录，假设没有依赖项。")
            classpath_compile = ""
            classpath_run = f"{compile_output_dir}"  # 运行时只包含 bin 目录

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
            print(f"写入 Java 文件失败: {e}", file=sys.stderr)
            return None

        # 【改进】使用 glob 收集所有 .java 文件，实现跨平台
        try:
            java_files = glob.glob(os.path.join(src_path, "**/*.java"), recursive=True)
            if not java_files:
                print(f"错误：在 {src_path} 中没有找到任何 .java 源文件。", file=sys.stderr)
                return None

            with open(sources_file, 'w', encoding='utf-8') as fs:
                fs.write("\n".join(java_files))
        except Exception as e:
            print(f"收集 Java 源文件失败: {e}", file=sys.stderr)
            return None

        # 编译命令
        compile_cmd = [
            "javac",
            "-d", compile_output_dir,  # 【补完】输出目录
            "-sourcepath", src_path,  # 【新增】源码根目录
            # f"@{sources_file}"              # 旧代码
        ]
        if classpath_compile:
            compile_cmd.extend(["-cp", classpath_compile])
        compile_cmd.append(f"@{sources_file}")  # 把 @sources_file 放到最后

        # 运行编译 (这是一个单独的 subprocess 调用，是允许的)
        compile_process = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True
        )

        try:
            if compile_process.returncode != 0:
                print(f"编译失败! STDERR: {compile_process.stderr}", file=sys.stderr)
                return None  # 编译失败，返回极差分数
            else:
                # print("Compilation succeeded!") # 成功，安静处理
                pass
        except Exception as e:
            print(f"编译检查出错: {e}", file=sys.stderr)
            return None

        # --- 4. 【修改】串行评估 ---
        try:
            # 【重要修改】
            # 移除 joblib.Parallel，改为标准的 for 循环
            # 这可以避免在子进程中创建新的子进程池

            print(f"子进程 {subprocess_index} 开始串行评估 {len(instance_commands)} 个实例...")
            results = []
            for cmd in instance_commands:
                # 在循环中一个一个地、串行地调用 self.run_command
                # self.run_command 内部会调用 subprocess.run
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

            print(f"子进程 {subprocess_index} 评估完成。平均适应度: {final_fitness}")
            return final_fitness

        except Exception as e:
            print(f"串行评估失败: {e}", file=sys.stderr)
            return None

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        # 假设我们总是评估第一个子进程（用于测试）
        # 在实际的进化框架中，subprocess_index 会由框架传入
        return self.evaluate(program_str, subprocess_index=0)


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

    # 【重要】你需要确保 aim_java_relative_path 指向的文件
    # (例如 "Method/AILS-II/src/SearchMethod/ParamAdjust.java")
    # 确实应该包含上面的 package 和 class。
    # 请确保 LLM 生成的代码与 aim_java_relative_path 的文件名和包名一致。

    eval = Ails2Evaluation(timeout_seconds=60)  # 设置超时

    # 假设你的 aim_java_relative_path 是 "Method/AILS-II/src/SearchMethod/ParamAdjust.java"
    # 那么你的 java_script 必须包含 "package SearchMethod;" 和 "public class ParamAdjust { ... }"

    # ！！！！！
    # 运行此测试前，请确保：
    # 1. java_dir (CVRPLIB-2025-AILSII) 目录已存在
    # 2. 它已经被复制为 CVRPLIB-2025-AILSII_0
    # 3. 你在代码中填写了所有 【!!!】 标记的路径
    # 4. CVRPLIB-2025-AILSII_0 目录下有 'data/*.vrp', 'libs/*', 'Method/AILS-II/src/**/*.java'
    # 5. template.py 中的 aim_java_relative_path 设置正确
    # ！！！！！

    print("开始评估...")
    res = eval.evaluate_program(java_script, None)
    print(f"评估完成。平均适应度: {res}")

