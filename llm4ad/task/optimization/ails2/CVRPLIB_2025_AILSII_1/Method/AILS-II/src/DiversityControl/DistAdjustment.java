package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long ini;
    private final double executionMaximumLimit;
    private double alpha = 1;
    private final StoppingCriterionType stoppingCriterionType;
    private final IdealDist idealDist;
    private DecayFunction decayFunction; // Flexible decay function implementation

    // DecayFunction interface for adaptive decay schedules
    interface DecayFunction {
        double calculateAlpha(int currentIteration, int totalIterations, double executionLimit);
    }

    // Exponential decay function implementation
    private static class ExponentialDecay implements DecayFunction {
        @Override
        public double calculateAlpha(int currentIteration, int totalIterations, double executionLimit) {
            return Math.exp(-((double) currentIteration / executionLimit));
        }
    }

    // Cosine decay function implementation
    private static class CosineDecay implements DecayFunction {
        @Override
        public double calculateAlpha(int currentIteration, int totalIterations, double executionLimit) {
            return 0.5 * (1 + Math.cos(Math.PI * currentIteration / executionLimit));
        }
    }

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit, DecayFunction decayFunction) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayFunction = decayFunction; // Assigning decay function
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }

        iterator++;

        // Calls the appropriate adjustment based on stoppingCriterionType
        switch (stoppingCriterionType) {
            case Iteration:
                alpha = iterationAdjustment();
                break;
            case Time:
                alpha = timeAdjustment();
                break;
            default:
                break;
        }

        // Apply the decay factor and clamp the ideal distance
        idealDist.idealDist *= alpha;
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }

    // Iterative adjustment utilizing the chosen decay function
    private double iterationAdjustment() {
        return decayFunction.calculateAlpha(iterator, distMMax, executionMaximumLimit);
    }

    // Time-based adjustment utilizing the chosen decay function
    private double timeAdjustment() {
        double current = (double) (System.currentTimeMillis() - ini) / 1000;
        double timePercentage = current / executionMaximumLimit;
        int currentIter = Math.min(iterator, (int)(timePercentage * executionMaximumLimit)); // Clamp iterations
        return decayFunction.calculateAlpha(currentIter, (int)executionMaximumLimit, executionMaximumLimit);
    }
}