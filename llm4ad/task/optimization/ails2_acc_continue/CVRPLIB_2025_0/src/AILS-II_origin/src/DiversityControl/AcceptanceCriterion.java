package DiversityControl;

import Auxiliary.Mean;
import Data.Instance;
import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;
import Solution.Solution;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

public class AcceptanceCriterion {
    private double thresholdOF;
    private double eta, etaMin, etaMax;
    private long startTime;
    private double alpha = 1;
    private int globalIterator = 0;
    private final StoppingCriterionType stoppingCriterionType;
    private final double executionMaximumLimit;
    private final int numIterUpdate;
    private double upperLimit = Integer.MAX_VALUE, updatedUpperLimit = Integer.MAX_VALUE;
    private final Mean averageLSfunction;
    private final ThreadMXBean threadMXBean;

    public AcceptanceCriterion(Instance instance, Config config, Double executionMaximumLimit) {
        this.etaMax = config.getEtaMax();
        this.etaMin = config.getEtaMin();
        this.eta = etaMax; // Initialize eta to max
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.averageLSfunction = new Mean(config.getGamma());
        this.numIterUpdate = config.getGamma();
        this.executionMaximumLimit = executionMaximumLimit;
        this.threadMXBean = ManagementFactory.getThreadMXBean();
    }

    public boolean acceptSolution(Solution solution) {
        if (globalIterator == 0) {
            startTime = System.nanoTime();
        }

        averageLSfunction.setValue(solution.f);
        globalIterator++;

        updateLimits(solution);

        if (isUpdateIteration()) {
            updateAlpha();
        }

        eta = Math.max(etaMin, eta * alpha);
        thresholdOF = computeThresholdOF();

        return solution.f <= thresholdOF;
    }

    private void updateLimits(Solution solution) {
        if (solution.f < updatedUpperLimit) {
            updatedUpperLimit = solution.f;
        }
        if (solution.f < upperLimit) {
            upperLimit = solution.f;
        }
    }

    private boolean isUpdateIteration() {
        return globalIterator % numIterUpdate == 0;
    }

    private void updateAlpha() {
        if (stoppingCriterionType == StoppingCriterionType.Iteration) {
            alpha = Math.pow(etaMin / etaMax, 1.0 / executionMaximumLimit);
        } else if (stoppingCriterionType == StoppingCriterionType.Time) {
            double currentTime = (double) (System.nanoTime() - startTime) / 1_000_000_000; // Time in seconds
            double timePercentage = currentTime / executionMaximumLimit;
            double totalIterations = (double) globalIterator / timePercentage;
            alpha = Math.pow(etaMin / etaMax, 1.0 / totalIterations);
        }
        // Reset upper limit after updates
        if (isUpdateIteration()) {
            upperLimit = updatedUpperLimit;
            updatedUpperLimit = Integer.MAX_VALUE;
        }
    }

    private double computeThresholdOF() {
        return upperLimit + (eta * (averageLSfunction.getDynamicAverage() - upperLimit));
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