package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private double distMMin;
    private double distMMax;
    private int iterator;
    private long ini;
    private final double executionMaximumLimit;
    private double alpha = 1;
    private final StoppingCriterionType stoppingCriterionType;
    private final IdealDist idealDist;
    
    // DecayFunction interface for adaptive decay schedules
    interface DecayFunction {
        double calculateAlpha(int currentIteration, double executionLimit);
    }

    // Default decay function implementation using exponential decay
    class ExponentialDecay implements DecayFunction {
        @Override
        public double calculateAlpha(int currentIteration, double executionLimit) {
            return Math.exp(-currentIteration / executionLimit);
        }
    }

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax; // Start at maximum ideal distance
        this.stoppingCriterionType = config.getStoppingCriterionType();
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }

        iterator++;

        // Calls the appropriate adjustment based on stoppingCriterionType
        switch (stoppingCriterionType) {
            case Iteration:
                alpha = new ExponentialDecay().calculateAlpha(iterator, (int) executionMaximumLimit);
                break;
            case Time:
                double currentTime = (double) (System.currentTimeMillis() - ini) / 1000;
                double timePercentage = currentTime / executionMaximumLimit;
                alpha = new ExponentialDecay().calculateAlpha((int) (timePercentage * iterator), (int) executionMaximumLimit);
                break;
            default:
                break;
        }

        // Apply the decay factor and clamp the ideal distance
        idealDist.idealDist *= alpha;
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }
}