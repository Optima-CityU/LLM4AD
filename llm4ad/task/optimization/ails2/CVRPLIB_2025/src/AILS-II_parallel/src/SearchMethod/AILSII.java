/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package SearchMethod;

import java.lang.reflect.InvocationTargetException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Random;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;

import Auxiliary.Distance;
import Data.Instance;
import DiversityControl.DistAdjustment;
import DiversityControl.OmegaAdjustment;
import DiversityControl.AcceptanceCriterion;
import DiversityControl.IdealDist;
import Improvement.LocalSearch;
import Improvement.IntraLocalSearch;
import Improvement.FeasibilityPhase;
import Perturbation.Perturbation;
import Perturbation.InsertionHeuristic;
import Solution.Solution;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ExecutionException;


public class AILSII 
{
    // ================ 修改开始：添加内部类 ================
    /**
     * 一个简单的内部类，用于封装来自并行任务的返回结果。
     * 它包含最终的解，以及该任务消耗的 CPU 时间（纳秒）。
     */
    private static class TaskResult {
        final Solution solution;
        final long cpuTime; // 纳秒

        TaskResult(Solution solution, long cpuTime) {
            this.solution = solution;
            this.cpuTime = cpuTime;
        }
    }
    // ================ 修改结束 ================

	//----------Problema------------
	Solution solution,referenceSolution,bestSolution;
	
	Instance instance;
	Distance pairwiseDistance;
	double bestF=Double.MAX_VALUE;
	double executionMaximumLimit;
	double optimal;
	
	//----------caculoLimiar------------
	int numIterUpdate;

	//----------Metricas------------
	int iterator,iteratorMF;
	long first,ini;
	double timeAF,totalTime,time;
	ThreadMXBean threadMXBean;

    private long totalWorkerCpuTime = 0L;
	
	Random rand=new Random();
	
	HashMap<String,OmegaAdjustment>omegaSetup=new HashMap<String,OmegaAdjustment>();

	double distanceLS;
	
	Perturbation[] pertubOperators;
	Perturbation selectedPerturbation;
	
	FeasibilityPhase feasibilityOperator;
	ConstructSolution constructSolution;
	
	LocalSearch localSearch;

	InsertionHeuristic insertionHeuristic;
	IntraLocalSearch intraLocalSearch;
	AcceptanceCriterion acceptanceCriterion;
//	----------Mare------------
	DistAdjustment distAdjustment;
//	---------Print----------
	boolean print=true;
	IdealDist idealDist;
	
	double epsilon;
	DecimalFormat deci=new DecimalFormat("0.0000");
	StoppingCriterionType stoppingCriterionType;
	
	// ================ 修改开始：添加文件输出功能 ================
	//----------文件输出------------
	private String outputDirectory = "Results/"; // 设置输出目录
	private boolean outputAllSolutions = false; // 控制是否输出所有解，默认为false（只输出最终解）
	private String instanceName = "default"; // 存储实例名称
	private FileWriter csvWriter; // CSV文件写入器
	// ================ 修改结束 ================

    private ExecutorService executorService;
    private Config config; // 存储Config以供线程使用

	public AILSII(Instance instance,InputParameters reader)
	{ 
		this.instance=instance;
		Config config=reader.getConfig();

        this.config = config; // 将局部的config存到类字段中
        this.executorService = Executors.newFixedThreadPool(2); // 创建一个固定大小为2的线程池

		this.optimal=reader.getBest();
		this.executionMaximumLimit=reader.getTimeLimit();
		this.threadMXBean = ManagementFactory.getThreadMXBean();
		
		this.epsilon=config.getEpsilon();
		this.stoppingCriterionType=config.getStoppingCriterionType();
		this.idealDist=new IdealDist();
		this.solution =new Solution(instance,config);
		this.referenceSolution =new Solution(instance,config);
		this.bestSolution =new Solution(instance,config);
		this.numIterUpdate=config.getGamma();
		
		this.pairwiseDistance=new Distance();
		
		this.pertubOperators=new Perturbation[config.getPerturbation().length];
		
		this.distAdjustment=new DistAdjustment( idealDist, config, executionMaximumLimit);
		
		this.intraLocalSearch=new IntraLocalSearch(instance,config);
		
		this.localSearch=new LocalSearch(instance,config,intraLocalSearch);
		
		this.feasibilityOperator=new FeasibilityPhase(instance,config,intraLocalSearch);
		
		this.constructSolution=new ConstructSolution(instance,config);
		
		OmegaAdjustment newOmegaAdjustment;
		for (int i = 0; i < config.getPerturbation().length; i++) 
		{
			newOmegaAdjustment=new OmegaAdjustment(config.getPerturbation()[i], config,instance.getSize(),idealDist);
			omegaSetup.put(config.getPerturbation()[i]+"", newOmegaAdjustment);
		}
		
		this.acceptanceCriterion=new AcceptanceCriterion(instance,config,executionMaximumLimit);

		try 
		{
			for (int i = 0; i < pertubOperators.length; i++) 
			{
				this.pertubOperators[i]=(Perturbation) Class.forName("Perturbation."+config.getPerturbation()[i]).
				getConstructor(Instance.class,Config.class,HashMap.class,IntraLocalSearch.class).
				newInstance(instance,config,omegaSetup,intraLocalSearch);
			}
			
		} catch (InstantiationException | IllegalAccessException | IllegalArgumentException
				| InvocationTargetException | NoSuchMethodException | SecurityException
				| ClassNotFoundException e) {
			e.printStackTrace();
		}
		
		// ================ 修改开始：初始化输出目录 ================
		// 初始化输出目录
		initializeOutputDirectory();
		// ================ 修改结束 ================
		
	}

	// ================ 修改开始：添加新方法 ================
	/**
	 * 初始化输出目录
	 */
	private void initializeOutputDirectory() {
		try {
			java.io.File directory = new java.io.File(outputDirectory);
			if (!directory.exists()) {
				directory.mkdirs();
			}
		} catch (Exception e) {
			System.err.println("创建输出目录失败: " + e.getMessage());
		}
	}

	/**
	 * 初始化CSV文件
	 */
	private void initializeCSVFile() {
		try {
			String csvFilePath = outputDirectory + instanceName + ".csv";
			csvWriter = new FileWriter(csvFilePath);
			System.out.println("CSV文件已创建: " + csvFilePath);
		} catch (IOException e) {
			System.err.println("创建CSV文件失败: " + e.getMessage());
		}
	}

	/**
	 * 写入CSV数据
	 * @param time 时间
	 * @param bestF 最优成本
	 */
	private void writeToCSV(double time, double bestF) {
		if (csvWriter != null) {
			try {
				csvWriter.write(deci.format(time).replace(",", ".") + ";" + deci.format(bestF).replace(",", ".") + "\n");
				csvWriter.flush();
			} catch (IOException e) {
				System.err.println("写入CSV文件失败: " + e.getMessage());
			}
		}
	}

	/**
	 * 关闭CSV文件
	 */
	private void closeCSVFile() {
		if (csvWriter != null) {
			try {
				csvWriter.close();
				System.out.println("CSV文件已关闭");
			} catch (IOException e) {
				System.err.println("关闭CSV文件失败: " + e.getMessage());
			}
		}
	}

	/**
	 * 获取指定路线的节点序列
	 * @param routeIndex 路线索引
	 * @return 节点序列字符串
	 */
	private String getRouteNodes(int routeIndex) {
		// 使用 Route 类的 toString2() 方法获取路线信息
		String routeString = bestSolution.routes[routeIndex].toString2();
		
		// 去掉开头的 "Route #X: " 部分，只保留节点序列
		int colonIndex = routeString.indexOf(":");
		if (colonIndex != -1 && colonIndex + 1 < routeString.length()) {
			return routeString.substring(colonIndex + 1).trim();
		}
		return routeString.trim();
	}

	/**
	 * 生成安全的文件名（只使用时间）
	 * @param time 时间
	 * @return 安全的文件名
	 */
	private String generateSafeFilename(double time) {
        // 使用实例名作为文件名，如果为空则使用默认名
        String name = (instanceName == null || instanceName.trim().isEmpty()) ? "instance" : instanceName.trim();

        // 移除文件名中的非法字符
        name = name.replaceAll("[\\\\/:*?\"<>|]", "_");

        // 确保以 .sol 结尾
        if (!name.toLowerCase().endsWith(".sol")) {
            name = name + ".sol";
        }
        return name;
	}

	/**
	 * 将解决方案写入文件
	 * @param currentBestF 当前最优成本
	 * @param currentTime 当前时间
	 */
	private void writeSolutionToFile(double currentBestF, double currentTime) {
		String filename = generateSafeFilename(currentTime);
		String fullPath = outputDirectory + filename;
		
		try (FileWriter writer = new FileWriter(fullPath)) {
			// 写入标准的CVRP sol格式
			// 先写所有路线
			for (int i = 0; i < bestSolution.numRoutes; i++) {
				String routeNodes = getRouteNodes(i);
				writer.write("Route #" + (i + 1) + ": " + routeNodes + "\n");
			}
			
			// 最后写入成本和时间
			writer.write("Cost " + deci.format(currentBestF).replace(",", ".") + "\n");
			writer.write("Time " + deci.format(currentTime).replace(",", "."));
			
			writer.flush();
			if (print) {
				System.out.println("Solution saved to: " + fullPath);
			}
		} catch (IOException e) {
			System.err.println("写入解决方案文件失败: " + e.getMessage());
		} catch (Exception e) {
			System.err.println("写入文件时发生错误: " + e.getMessage());
		}
	}

	/**
	 * 将最终解决方案写入文件（使用最终时间作为名称）
	 */
	private void writeFinalSolutionToFile() {
		// 使用最终时间作为文件名
        String filename = generateSafeFilename(totalTime);
        String fullPath = outputDirectory + filename;
		
		try (FileWriter writer = new FileWriter(fullPath)) {
			// 写入标准的CVRP sol格式
			// 先写所有路线
			for (int i = 0; i < bestSolution.numRoutes; i++) {
				String routeNodes = getRouteNodes(i);
				writer.write("Route #" + (i + 1) + ": " + routeNodes + "\n");
			}
			
			// 最后写入成本
			writer.write("Cost " + deci.format(bestF).replace(",", "."));
			
			writer.flush();
			if (print) {
				System.out.println("Final solution saved to: " + filename);
				System.out.println("Final solution cost: " + deci.format(bestF).replace(",", "."));
			}
		} catch (IOException e) {
			System.err.println("写入最终解决方案文件失败: " + e.getMessage());
		}
	}
	// ================ 修改结束 ================

	public void search() {
		iterator = 0;
		first = threadMXBean.getCurrentThreadCpuTime(); // 获取当前线程的CPU时间

        totalWorkerCpuTime = 0L;

		referenceSolution.numRoutes = instance.getMinNumberRoutes();
		constructSolution.construct(referenceSolution);

		feasibilityOperator.makeFeasible(referenceSolution);
		localSearch.localSearch(referenceSolution, true);
		
		// ================ 修改开始：修复初始化问题 ================
		// 先设置bestSolution
		bestSolution.clone(referenceSolution);
		bestF = bestSolution.f; // 使用实际的解成本，而不是Double.MAX_VALUE
		
		// ================ 修改开始：根据配置决定是否保存初始解 ================
		// 如果配置为输出所有解，则保存初始解
		if (outputAllSolutions) {
			double initialTime = (double) (threadMXBean.getCurrentThreadCpuTime() - first) / 1_000_000_000;
			if (print) {
				System.out.println("Initial solution: " + initialTime + ";" + bestF);
			}
			writeSolutionToFile(bestF, initialTime);
		}
		// ================ 修改结束 ================

        while (!stoppingCriterion()) {
            iterator++;

            // ================ 修改开始：并行扰动和搜索 ================

            // 1. 定义两个线程任务 (Callable<TaskResult>)

            Callable<TaskResult> task1 = () -> {
                long taskStartTime = threadMXBean.getCurrentThreadCpuTime(); // 任务1的CPU计时器

                // 线程安全：创建此线程专属的 Solution 和搜索工具
                Solution threadSolution = new Solution(instance, config);
                threadSolution.clone(referenceSolution);
                IntraLocalSearch threadIntraLS = new IntraLocalSearch(instance, config);
                LocalSearch threadLS = new LocalSearch(instance, config, threadIntraLS);
                FeasibilityPhase threadFP = new FeasibilityPhase(instance, config, threadIntraLS);

                // 应用扰动和搜索
                pertubOperators[0].applyPerturbation(threadSolution);
                threadFP.makeFeasible(threadSolution);
                threadLS.localSearch(threadSolution, true);

                long taskEndTime = threadMXBean.getCurrentThreadCpuTime(); // 任务1的CPU计时器结束
                long taskCpuTime = taskEndTime - taskStartTime;
                return new TaskResult(threadSolution, taskCpuTime); // 返回结果和CPU时间
            };

            Callable<TaskResult> task2 = () -> {
                long taskStartTime = threadMXBean.getCurrentThreadCpuTime(); // 任务2的CPU计时器

                // 线程安全：创建此线程专属的 Solution 和搜索工具
                Solution threadSolution = new Solution(instance, config);
                threadSolution.clone(referenceSolution);
                IntraLocalSearch threadIntraLS = new IntraLocalSearch(instance, config);
                LocalSearch threadLS = new LocalSearch(instance, config, threadIntraLS);
                FeasibilityPhase threadFP = new FeasibilityPhase(instance, config, threadIntraLS);

                // 应用扰动和搜索
                pertubOperators[1].applyPerturbation(threadSolution);
                threadFP.makeFeasible(threadSolution);
                threadLS.localSearch(threadSolution, true);

                long taskEndTime = threadMXBean.getCurrentThreadCpuTime(); // 任务2的CPU计时器结束
                long taskCpuTime = taskEndTime - taskStartTime;
                return new TaskResult(threadSolution, taskCpuTime); // 返回结果和CPU时间
            };

            // 2. 提交任务 (Future<TaskResult>)
            Future<TaskResult> future1 = executorService.submit(task1);
            Future<TaskResult> future2 = executorService.submit(task2);

            TaskResult result1 = null;
            TaskResult result2 = null;

            // 3. 获取结果
            try {
                result1 = future1.get();
                result2 = future2.get();
            } catch (InterruptedException | ExecutionException e) {
                System.err.println("一个扰动线程执行失败: " + e.getMessage());
                e.printStackTrace();
                continue;
            }

            // 4. 将两个工作线程的 CPU 时间计入总和
            // 这是“总 CPU 时间”方案
//            totalWorkerCpuTime += result1.cpuTime;
//            totalWorkerCpuTime += result2.cpuTime;

            totalWorkerCpuTime += Math.max(result1.cpuTime, result2.cpuTime);

            /* * 备选方案：如果你*真的*想要 "CPU makespan" (一种非标准度量)
             * totalWorkerCpuTime += Math.max(result1.cpuTime, result2.cpuTime);
             * (我强烈不推荐这样做，但这实现了你字面上的“makespan”要求)
             */


            // 5. 比较两个结果，选出最好的
            Solution solution1 = result1.solution;
            Solution solution2 = result2.solution;
            Solution bestOfTwo;
            Perturbation perturbationUsed;

            if (solution1.f < solution2.f) {
                bestOfTwo = solution1;
                perturbationUsed = pertubOperators[0];
            } else {
                bestOfTwo = solution2;
                perturbationUsed = pertubOperators[1];
            }

            // 6. 像以前一样继续...
            solution.clone(bestOfTwo);
            distanceLS = pairwiseDistance.pairwiseSolutionDistance(solution, referenceSolution);
            evaluateSolution();
            distAdjustment.distAdjustment();
            perturbationUsed.getChosenOmega().setDistance(distanceLS);
            if (acceptanceCriterion.acceptSolution(solution))
                referenceSolution.clone(solution);

            // ================ 修改结束 ================
        }

        // ================ 修改开始：修改总时间计算 ================
        long mainCpuTime = threadMXBean.getCurrentThreadCpuTime() - first;
        long totalCpuTime = mainCpuTime + totalWorkerCpuTime; // 主线程 + 所有工作线程
        totalTime = (double) totalCpuTime / 1_000_000_000.0; // 转换为秒
        // ================ 修改结束 ================

        // ================ 修改开始：在搜索结束后保存最终解 ================

        // !! 在保存文件之前，关闭线程池 !!
        executorService.shutdown();
        try {
            // 等待最多60秒让正在运行的任务结束
            if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                executorService.shutdownNow(); // 强制关闭
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
        }

        // 保存最终的最优解（使用总时间作为文件名）
        writeFinalSolutionToFile();

        // 关闭CSV文件
        closeCSVFile();
        // ================ 修改结束 ================
    }

    public void evaluateSolution() {
        if ((solution.f - bestF) < -epsilon) {
            // ================ 修改开始：修复clone前的空指针问题 ================
            // 先检查solution是否有效
            if (solution != null && solution.routes != null) {
                bestF = solution.f;
                bestSolution.clone(solution);
                iteratorMF = iterator;

                // ================ 修改开始：使用总 CPU 时间 ================

                // 1. 获取主线程*当前*消耗的 CPU 时间 (从 'first' 开始)
                long mainCpuTime = threadMXBean.getCurrentThreadCpuTime() - first;

                // 2. 加上我们一直在 search() 循环中累积的*工作线程*消耗的 CPU 时间
                //    (totalWorkerCpuTime 是一个类成员字段)
                long totalCpuTime = mainCpuTime + totalWorkerCpuTime;

                // 3. 将总的 CPU 时间（纳秒）转换为秒，这现在是你的“混合CPU时间”
                timeAF = (double) totalCpuTime / 1_000_000_000.0;

                // ================ 修改结束 ================

                if (print) {
                    System.out.println(timeAF + ";" + bestF);
                }

                // ================ 修改开始：写入CSV文件 ================
                // 每次找到更好的解时都写入CSV文件
                writeToCSV(timeAF, bestF);
                // ================ 修改结束 ================

                // ================ 修改开始：根据配置决定是否每次保存文件 ================
                // 如果配置为输出所有解，则每次找到更好的解时都保存文件
                if (outputAllSolutions) {
                    writeSolutionToFile(bestF, timeAF);
                }
                // ================ 修改结束 ================
            }
            // ================ 修改结束 ================
        }
    }

    private boolean stoppingCriterion() {
        switch (stoppingCriterionType) {
            case Iteration:
                // 迭代次数的判断保持不变
                if (bestF <= optimal || executionMaximumLimit <= iterator)
                    return true;
                break;

            case Time:
                // ================ 修改开始 ================

                // 1. 获取主线程*当前*消耗的 CPU 时间 (从 'first' 开始)
                long mainCpuTime = threadMXBean.getCurrentThreadCpuTime() - first;

                // 2. 加上我们一直在循环中累积的*工作线程*消耗的 CPU 时间
                long totalCpuTime = mainCpuTime + totalWorkerCpuTime;

                // 3. 将总的 CPU 时间（纳秒）转换为秒
                double elapsedCpuSeconds = (double) totalCpuTime / 1_000_000_000.0;

                // 4. 使用这个“总 CPU 时间”来和你的限制 (executionMaximumLimit) 比较
                if (bestF <= optimal || executionMaximumLimit < elapsedCpuSeconds)
                    return true;

                // ================ 修改结束 ================
                break;
        }
        return false;
    }

//	private boolean stoppingCriterion() {
//		switch (stoppingCriterionType) {
//			case Iteration:
//				if (bestF <= optimal || executionMaximumLimit <= iterator)
//					return true;
//				break;
//
//			case Time:
//				if (bestF <= optimal || executionMaximumLimit < (threadMXBean.getCurrentThreadCpuTime() - first) / 1_000_000_000)
//					return true;
//				break;
//		}
//		return false;
//	}
	
	// ================ 修改开始：添加实例名称设置方法 ================
	public void setInstanceName(String instanceName) {
		// 根据实例名称更新输出目录
		this.outputDirectory = "Results/" + instanceName + "/";
		this.instanceName = instanceName;
		// 重新初始化输出目录
		initializeOutputDirectory();
		// 初始化CSV文件
		initializeCSVFile();
		System.out.println("输出目录设置为: " + this.outputDirectory);
	}
	// ================ 修改结束 ================
	
	public static void main(String[] args) 
	{
		InputParameters reader = new InputParameters();
		reader.readingInput(args);
		
		Instance instance = new Instance(reader);
		
		AILSII ailsII = new AILSII(instance, reader);
		
		// ================ 修改开始：从命令行参数获取实例名称 ================
		String instanceName = "default";
		// 尝试从命令行参数中提取实例名称
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-file") && i + 1 < args.length) {
				String filePath = args[i + 1];
				File file = new File(filePath);
				String fileName = file.getName();
				if (fileName.lastIndexOf(".") > 0) {
					instanceName = fileName.substring(0, fileName.lastIndexOf("."));
				} else {
					instanceName = fileName;
				}
				break;
			}
		}
		
		// 设置实例名称（这会自动创建对应的输出目录）
		ailsII.setInstanceName(instanceName);
		
		// 设置输出模式
		boolean outputAllSteps = false; // 这里设置为false只输出最终解
		ailsII.setOutputAllSolutions(outputAllSteps);
		
		if (outputAllSteps) {
			System.out.println("输出模式: 每一步都输出解文件 - 实例: " + instanceName);
		} else {
			System.out.println("输出模式: 只输出最终解 - 实例: " + instanceName);
		}
		// ================ 修改结束 ================
		
		ailsII.search();
	}
	
	public Solution getBestSolution() {
		return bestSolution;
	}

	public double getBestF() {
		return bestF;
	}

	public double getGap()
	{
		return 100*((bestF-optimal)/optimal);
	}
	
	public boolean isPrint() {
		return print;
	}

	public void setPrint(boolean print) {
		this.print = print;
	}

	// ================ 修改开始：添加输出模式控制方法 ================
	public boolean isOutputAllSolutions() {
		return outputAllSolutions;
	}

	public void setOutputAllSolutions(boolean outputAllSolutions) {
		this.outputAllSolutions = outputAllSolutions;
	}
	// ================ 修改结束 ================
	
	public Solution getSolution() {
		return solution;
	}

	public int getIterator() {
		return iterator;
	}

	public String printOmegas()
	{
		String str="";
		for (int i = 0; i < pertubOperators.length; i++) 
		{
			str+="\n"+omegaSetup.get(this.pertubOperators[i].perturbationType+""+referenceSolution.numRoutes);
		}
		return str;
	}
	
	public Perturbation[] getPertubOperators() {
		return pertubOperators;
	}
	
	public double getTotalTime() {
		return totalTime;
	}
	
	public double getTimePerIteration() 
	{
		return totalTime/iterator;
	}

	public double getTimeAF() {
		return timeAF;
	}

	public int getIteratorMF() {
		return iteratorMF;
	}
	
	public double getConvergenceIteration()
	{
		return (double)iteratorMF/iterator;
	}
	
	public double convergenceTime()
	{
		return (double)timeAF/totalTime;
	}
	
}