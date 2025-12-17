/**
 * Copyright 2022, Vinícius R. Máximo
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


public class AILSII
{
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

    //----------文件输出配置------------
    private String outputDirectory = "Results/"; // 默认输出目录
    private boolean customOutputSet = false;     // 标记是否使用了自定义输出目录
    private String instanceName = "default";     // 存储实例名称
    private FileWriter csvWriter;                // CSV文件写入器

    public AILSII(Instance instance,InputParameters reader)
    {
        this.instance=instance;
        Config config=reader.getConfig();
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

        // 初始化放在 setInstanceName 或 setOutputDirectory 中处理
    }

    /**
     * 设置输出目录 (新添加的方法)
     * @param path 输出路径
     */
    public void setOutputDirectory(String path) {
        if (path != null && !path.isEmpty()) {
            this.outputDirectory = path;
            if (!this.outputDirectory.endsWith(File.separator)) {
                this.outputDirectory += File.separator;
            }
            this.customOutputSet = true;
            initializeOutputDirectory();
            // 如果此时已经有 instanceName，可以尝试初始化文件
        }
    }

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
            // System.out.println("CSV文件已创建: " + csvFilePath);
        } catch (IOException e) {
            System.err.println("创建CSV文件失败: " + e.getMessage());
        }
    }

    /**
     * 写入CSV数据
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
            } catch (IOException e) {
                System.err.println("关闭CSV文件失败: " + e.getMessage());
            }
        }
    }

    private String getRouteNodes(int routeIndex) {
        String routeString = bestSolution.routes[routeIndex].toString2();
        int colonIndex = routeString.indexOf(":");
        if (colonIndex != -1 && colonIndex + 1 < routeString.length()) {
            return routeString.substring(colonIndex + 1).trim();
        }
        return routeString.trim();
    }

    private String getFixedFilename() {
        String name = (instanceName == null || instanceName.trim().isEmpty()) ? "instance" : instanceName.trim();
        if (!name.toLowerCase().endsWith(".sol")) {
            name = name + ".sol";
        }
        return name;
    }

    private void writeSolutionToFile(double currentBestF, double currentTime) {
        String filename = getFixedFilename();
        String fullPath = outputDirectory + filename;

        try (FileWriter writer = new FileWriter(fullPath, false)) {
            for (int i = 0; i < bestSolution.numRoutes; i++) {
                String routeNodes = getRouteNodes(i);
                writer.write("Route #" + (i + 1) + ": " + routeNodes + "\n");
            }
            writer.write("Cost " + deci.format(currentBestF).replace(",", ".") + "\n");
            writer.write("Time " + deci.format(currentTime).replace(",", "."));
            writer.flush();
        } catch (IOException e) {
            System.err.println("写入解决方案文件失败: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("写入文件时发生错误: " + e.getMessage());
        }
    }

    public void search() {
        iterator = 0;
        first = threadMXBean.getCurrentThreadCpuTime();
        referenceSolution.numRoutes = instance.getMinNumberRoutes();
        constructSolution.construct(referenceSolution);

        feasibilityOperator.makeFeasible(referenceSolution);
        localSearch.localSearch(referenceSolution, true);

        bestSolution.clone(referenceSolution);
        bestF = bestSolution.f;

        double initialTime = (double) (threadMXBean.getCurrentThreadCpuTime() - first) / 1_000_000_000;
        if (print) {
            System.out.println("Initial solution: " + initialTime + ";" + bestF);
        }

        writeSolutionToFile(bestF, initialTime);

        while (!stoppingCriterion()) {
            iterator++;

            solution.clone(referenceSolution);

            selectedPerturbation = pertubOperators[0];
            selectedPerturbation.applyPerturbation(solution);
            feasibilityOperator.makeFeasible(solution);
            localSearch.localSearch(solution, true);
            distanceLS = pairwiseDistance.pairwiseSolutionDistance(solution, referenceSolution);

            evaluateSolution();
            distAdjustment.distAdjustment();

            selectedPerturbation.getChosenOmega().setDistance(distanceLS);

            if (acceptanceCriterion.acceptSolution(solution))
                referenceSolution.clone(solution);
        }

        totalTime = (double) (threadMXBean.getCurrentThreadCpuTime() - first) / 1_000_000_000;

        writeSolutionToFile(bestF, totalTime);
        closeCSVFile();
    }

    public void evaluateSolution() {
        if ((solution.f - bestF) < -epsilon) {
            if (solution != null && solution.routes != null) {
                bestF = solution.f;
                bestSolution.clone(solution);
                iteratorMF = iterator;
                timeAF = (double) (threadMXBean.getCurrentThreadCpuTime() - first) / 1_000_000_000;

                if (print) {
                    System.out.println(timeAF + ";" + bestF);
                }

                writeToCSV(timeAF, bestF);
                writeSolutionToFile(bestF, timeAF);
            }
        }
    }

    private boolean stoppingCriterion() {
        switch (stoppingCriterionType) {
            case Iteration:
                if (bestF <= optimal || executionMaximumLimit <= iterator)
                    return true;
                break;

            case Time:
                if (bestF <= optimal || executionMaximumLimit < (threadMXBean.getCurrentThreadCpuTime() - first) / 1_000_000_000)
                    return true;
                break;
        }
        return false;
    }

    public void setInstanceName(String instanceName) {
        this.instanceName = instanceName;

        // 如果没有手动设置输出路径，则保持原有的行为（创建以instanceName命名的子文件夹）
        if (!customOutputSet) {
            this.outputDirectory = "Results/" + instanceName + "/";
        }

        // 初始化目录和文件
        initializeOutputDirectory();
        initializeCSVFile();

        // System.out.println("输出目录: " + this.outputDirectory);
    }

    public static void main(String[] args)
    {
        InputParameters reader = new InputParameters();
        reader.readingInput(args);

        Instance instance = new Instance(reader);

        AILSII ailsII = new AILSII(instance, reader);

        String instanceName = "default";
        String customOutput = null;

        // 解析命令行参数
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
            }
            // [新增] 解析 output 参数
            if (args[i].equals("-output") && i + 1 < args.length) {
                customOutput = args[i + 1];
            }
        }

        // 优先设置自定义输出目录
        if (customOutput != null) {
            ailsII.setOutputDirectory(customOutput);
        }

        // 设置实例名称 (如果不设置 customOutput, 这里会使用默认的 Results/instanceName/ 结构)
        ailsII.setInstanceName(instanceName);

        System.out.println("开始搜索: " + instanceName);
        if (customOutput != null) {
            System.out.println("结果将保存至: " + customOutput);
        }

        ailsII.search();
    }

    // Getters/Setters 省略...
    public Solution getBestSolution() { return bestSolution; }
    public double getBestF() { return bestF; }
    public double getGap() { return 100*((bestF-optimal)/optimal); }
    public boolean isPrint() { return print; }
    public void setPrint(boolean print) { this.print = print; }
    public Solution getSolution() { return solution; }
    public int getIterator() { return iterator; }
    public String printOmegas() {
        String str="";
        for (int i = 0; i < pertubOperators.length; i++) {
            str+="\n"+omegaSetup.get(this.pertubOperators[i].perturbationType+""+referenceSolution.numRoutes);
        }
        return str;
    }
    public Perturbation[] getPertubOperators() { return pertubOperators; }
    public double getTotalTime() { return totalTime; }
    public double getTimePerIteration() { return totalTime/iterator; }
    public double getTimeAF() { return timeAF; }
    public int getIteratorMF() { return iteratorMF; }
    public double getConvergenceIteration() { return (double)iteratorMF/iterator; }
    public double convergenceTime() { return (double)timeAF/totalTime; }
}