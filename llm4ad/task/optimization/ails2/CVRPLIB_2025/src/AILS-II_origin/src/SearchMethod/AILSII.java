/**
 * Copyright 2022, Vinícius R. Máximo
 * Distributed under the terms of the MIT License.
 * SPDX-License-Identifier: MIT
 */
package SearchMethod;

import java.lang.reflect.InvocationTargetException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Random;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;
import java.io.FilenameFilter; // 引入过滤器

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
    //  ----------Mare------------
    DistAdjustment distAdjustment;
    //  ---------Print----------
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

    // 【新增】用于缓存本次运行确定的唯一文件名，防止在一次运行中多次生成不同后缀的文件
    private String currentUniqueFilename = null;

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
    }

    public void setOutputDirectory(String path) {
        if (path != null && !path.isEmpty()) {
            this.outputDirectory = path;
            if (!this.outputDirectory.endsWith(File.separator)) {
                this.outputDirectory += File.separator;
            }
            this.customOutputSet = true;
            initializeOutputDirectory();
        }
    }

    private void initializeOutputDirectory() {
        try {
            java.io.File directory = new java.io.File(outputDirectory);
            if (!directory.exists()) {
                directory.mkdirs();
            }
        } catch (Exception e) {
            System.err.println("create dir fail: " + e.getMessage());
        }
    }

    /**
     * 【修改】获取唯一文件名逻辑
     * 1. 扫描目录下所有以 "实例名_" 开头且以 ".sol" 结尾的文件
     * 2. 找到最大的数字后缀
     * 3. 返回 "实例名_(最大后缀+1).sol"
     * 4. 结果被缓存，确保本次运行期间一直写入同一个文件
     */
    private String getUniqueFilename() {
        // 如果本次运行已经确定了文件名，直接返回（避免每次发现新解都生成一个新后缀文件）
        if (currentUniqueFilename != null) {
            return currentUniqueFilename;
        }

        String baseName = (instanceName == null || instanceName.trim().isEmpty()) ? "instance" : instanceName.trim();
        // 去除可能存在的扩展名，确保baseName纯净
        if (baseName.toLowerCase().endsWith(".sol")) {
            baseName = baseName.substring(0, baseName.length() - 4);
        }

        File dir = new File(outputDirectory);
        int maxIndex = 0; // 默认最大索引为 0，这样如果没有文件，下一个就是 1

        if (dir.exists() && dir.isDirectory()) {
            final String prefix = baseName + "_";
            final String suffix = ".sol";

            // 过滤文件：必须以 "Name_" 开头，以 ".sol" 结尾
            File[] files = dir.listFiles(new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.startsWith(prefix) && name.toLowerCase().endsWith(suffix);
                }
            });

            if (files != null) {
                for (File f : files) {
                    String name = f.getName();
                    try {
                        // 提取中间的数字部分
                        // 例如: "X-n101-k25_12.sol" -> 提取 "12"
                        String numberPart = name.substring(prefix.length(), name.length() - suffix.length());
                        int index = Integer.parseInt(numberPart);
                        if (index > maxIndex) {
                            maxIndex = index;
                        }
                    } catch (NumberFormatException e) {
                        // 如果文件名符合模式但中间不是数字（例如 instance_best.sol），则忽略
                    }
                }
            }
        }

        // 下一个文件的后缀
        int nextIndex = maxIndex + 1;

        // 组合最终文件名: "instanceName_1.sol", "instanceName_2.sol" 等
        currentUniqueFilename = baseName + "_" + nextIndex + ".sol";

        // System.out.println("本次运行将结果保存至: " + currentUniqueFilename); // 可选：打印日志确认

        return currentUniqueFilename;
    }

    /**
     * 【修改】初始化CSV文件
     * 现在使用 getUniqueFilename() 来保持 .csv 文件名与 .sol 文件名后缀同步
     */
    private void initializeCSVFile() {
        try {
            // 1. 获取本次运行的唯一文件名 (例如 instance_1.sol)
            String uniqueSolName = getUniqueFilename();

            // 2. 将后缀 .sol 替换为 .csv (例如 instance_1.csv)
            // 注意：因为 getUniqueFilename 保证返回 .sol 结尾，所以可以直接替换
            String csvFilename = uniqueSolName.replace(".sol", ".csv");

            String csvFilePath = outputDirectory + csvFilename;
            csvWriter = new FileWriter(csvFilePath);
        } catch (IOException e) {
            System.err.println("Failed to create CSV file:" + e.getMessage());
        }
    }

    private void writeToCSV(double time, double bestF) {
        if (csvWriter != null) {
            try {
                csvWriter.write(deci.format(time).replace(",", ".") + ";" + deci.format(bestF).replace(",", ".") + "\n");
                csvWriter.flush();
            } catch (IOException e) {
                System.err.println("Failed to write to CSV file: " + e.getMessage());
            }
        }
    }

    private void closeCSVFile() {
        if (csvWriter != null) {
            try {
                csvWriter.close();
            } catch (IOException e) {
                System.err.println("Failed to close CSV file: " + e.getMessage());
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

    private void writeSolutionToFile(double currentBestF, double currentTime) {
        // 使用缓存好的唯一文件名
        String filename = getUniqueFilename();
        String fullPath = outputDirectory + filename;

        try (FileWriter writer = new FileWriter(fullPath, false)) { // false 表示覆盖模式，覆盖当前这个带后缀的文件
            for (int i = 0; i < bestSolution.numRoutes; i++) {
                String routeNodes = getRouteNodes(i);
                writer.write("Route #" + (i + 1) + ": " + routeNodes + "\n");
            }
            writer.write("Cost " + deci.format(currentBestF).replace(",", ".") + "\n");
            writer.write("Time " + deci.format(currentTime).replace(",", "."));
            writer.flush();
        } catch (IOException e) {
            System.err.println("Failed to write solution file: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("An error occurred while writing the file: " + e.getMessage());
        }
    }

    public void search() {
        iterator = 0;
        first = System.nanoTime();
        referenceSolution.numRoutes = instance.getMinNumberRoutes();
        constructSolution.construct(referenceSolution);

        feasibilityOperator.makeFeasible(referenceSolution);
        localSearch.localSearch(referenceSolution, true);

        bestSolution.clone(referenceSolution);
        bestF = bestSolution.f;

        double initialTime = (double) (System.nanoTime() - first) / 1_000_000_000;
        if (print) {
            System.out.println("Initial solution: " + initialTime + ";" + bestF);
        }

        writeSolutionToFile(bestF, initialTime);

        while (!stoppingCriterion()) {
            iterator++;

            solution.clone(referenceSolution);

            selectedPerturbation = pertubOperators[rand.nextInt(pertubOperators.length)];
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

        totalTime = (double) (System.nanoTime() - first) / 1_000_000_000;

        writeSolutionToFile(bestF, totalTime);
        closeCSVFile();
    }

    public void evaluateSolution() {
        if ((solution.f - bestF) < -epsilon) {
            if (solution != null && solution.routes != null) {
                bestF = solution.f;
                bestSolution.clone(solution);
                iteratorMF = iterator;
                timeAF = (double) (System.nanoTime() - first) / 1_000_000_000;

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
                if (bestF <= optimal || executionMaximumLimit < (System.nanoTime() - first) / 1_000_000_000)
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

        // 【注意】这里必须调用 initializeCSVFile 来创建带后缀的CSV文件
        initializeCSVFile();
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
            if (args[i].equals("-output") && i + 1 < args.length) {
                customOutput = args[i + 1];
            }
        }

        // 优先设置自定义输出目录
        if (customOutput != null) {
            ailsII.setOutputDirectory(customOutput);
        }

        // 设置实例名称
        ailsII.setInstanceName(instanceName);

        System.out.println("Start Search: " + instanceName);
        if (customOutput != null) {
            System.out.println("save results to: " + customOutput);
        }

        ailsII.search();
    }

    // Getters/Setters 保持不变
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