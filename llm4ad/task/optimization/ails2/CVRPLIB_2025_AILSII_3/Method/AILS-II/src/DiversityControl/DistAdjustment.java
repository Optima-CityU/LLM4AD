package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long startTime;
    private final double executionMaximumLimit;
    private double alpha = 1;
    private final StoppingCriterionType stoppingCriterionType;
    private final IdealDist idealDist;
    private DecayFunction decayFunction;

    // DecayFunction interface for adaptive decay schedules
    interface DecayFunction {
        double calculateAlpha(int currentIteration, double executionLimit);
    }

    // Default decay function implementation using piecewise linear decay
    static class DefaultDecayFunction implements DecayFunction {
        @Override
        public double calculateAlpha(int currentIteration, double executionLimit) {
            return (currentIteration < executionLimit) ? 
                Math.pow((double) currentIteration / executionLimit, 2) : 1.0;
        }
    }

    // Constructor allowing custom decay function
    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit, DecayFunction decayFunction) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayFunction = (decayFunction != null) ? decayFunction : new DefaultDecayFunction();
    }

    // Main method to adjust the ideal distance dynamically
    public void distAdjustment() {
        if (iterator == 0) {
            startTime = System.currentTimeMillis();
        }

        iterator++;

        // Determine alpha adjustment based on stopping criterion
        switch (stoppingCriterionType) {
            case Iteration:
                alpha = decayFunction.calculateAlpha(iterator, executionMaximumLimit);
                break;
            case Time:
                double elapsedTime = (double) (System.currentTimeMillis() - startTime) / 1000;
                alpha = decayFunction.calculateAlpha((int)(elapsedTime), (int)(executionMaximumLimit));
                break;
            default:
                break;
        }

        // Apply decay factor to ideal distance, ensuring it remains within bounds
        idealDist.idealDist *= alpha;
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }
}