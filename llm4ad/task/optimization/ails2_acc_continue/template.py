import os

template_program = '''
package DiversityControl;

import Auxiliary.Mean;
import Data.Instance;
import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;
import Solution.Solution;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

public class AcceptanceCriterion {
    double thresholdOF;
    double eta, etaMin, etaMax;
    long ini;
    double alpha = 1;
    int globalIterator = 0;
    StoppingCriterionType stoppingCriterionType;
    double executionMaximumLimit;
    int numIterUpdate;
    double upperLimit = Integer.MAX_VALUE, updatedUpperLimit = Integer.MAX_VALUE;
    Mean averageLSfunction;
    ThreadMXBean threadMXBean;

    public AcceptanceCriterion(Instance instance, Config config, Double executionMaximumLimit) {
        this.eta = config.getEtaMax();
        this.etaMin = config.getEtaMin();
        this.etaMax = config.getEtaMax();
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.averageLSfunction = new Mean(config.getGamma());
        this.numIterUpdate = config.getGamma();
        this.executionMaximumLimit = executionMaximumLimit;
        this.threadMXBean = ManagementFactory.getThreadMXBean(); //
    }

    public boolean acceptSolution(Solution solution) {
        if (globalIterator == 0) {
            ini = System.nanoTime(); //
        }

        averageLSfunction.setValue(solution.f);

        globalIterator++;

        if (globalIterator % numIterUpdate == 0) {
            upperLimit = updatedUpperLimit;
            updatedUpperLimit = Integer.MAX_VALUE;
        }

        if (solution.f < updatedUpperLimit) {
            updatedUpperLimit = solution.f;
        }

        if (solution.f < upperLimit) {
            upperLimit = solution.f;
        }

        // --------------------------------------------
        switch (stoppingCriterionType) {
            case Iteration:
                if (globalIterator % numIterUpdate == 0) {
                    alpha = Math.pow(etaMin / etaMax, (double) 1 / executionMaximumLimit);
                }
                break;

            case Time:
                if (globalIterator % numIterUpdate == 0) {
                    double maxTime = executionMaximumLimit;
                    double current = (double) (System.nanoTime() - ini) / 1_000_000_000; // 转换为秒
                    double timePercentage = current / maxTime;
                    double total = (double) globalIterator / timePercentage;

                    alpha = Math.pow(etaMin / etaMax, (double) 1 / total);
                }
                break;

            default:
                break;
        }

        eta *= alpha;
        eta = Math.max(eta, etaMin);

        thresholdOF = (int) (upperLimit + (eta * (averageLSfunction.getDynamicAverage() - upperLimit)));
        if (solution.f <= thresholdOF) {
            return true;
        } else {
            return false;
        }
    }

    public double getEta() {
        return eta;
    }

    public double getThresholdOF() {
        return thresholdOF;
    }

    public void setEta(double eta) {
        this.eta = eta;
    }

}
'''

task_description = """
Perform method-level improvements on the Java class `AcceptanceCriterion`.
This class acts as the core **Solution Acceptance Mechanism** within a metaheuristic optimization framework for the CVRP.
Your Ultimate Goal is to refactor this acceptance criterion to enhance the solver's ability to find high-quality solutions in complex CVRP instances.
"""

java_dir = "CVRPLIB_2025"        # 多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
aim_java_relative_path = os.path.join('src', 'AILS-II_origin','src', 'DiversityControl', 'AcceptanceCriterion.java')       # 被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"

# java_dir = "CVRPLIB_2025_AILSII"        # 多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
# aim_java_relative_path = os.path.join('Method', 'AILS-II','src', 'DiversityControl', 'AcceptanceCriterion.java')       # 被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"



