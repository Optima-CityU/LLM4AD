package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long ini;
    private final double executionMaximumLimit;
    private double alpha;
    private final StoppingCriterionType stoppingCriterionType;
    private final IdealDist idealDist;
    private final DecayFunction decayFunction;

    // DecayFunction interface for adaptive decay schedules
    interface DecayFunction {
        double calculateAlpha(int currentIteration, double executionLimit);
    }

    // Default decay function implementation (example: exponential decay)
    static class ExponentialDecayFunction implements DecayFunction {
        @Override
        public double calculateAlpha(int currentIteration, double executionLimit) {
            return Math.exp(-((double) currentIteration / executionLimit));
        }
    }

    // Additional decay function implementations can be added here (e.g., CosineDecayFunction)

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit, DecayFunction decayFunction) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayFunction = decayFunction != null ? decayFunction : new ExponentialDecayFunction(); // Default to exponential decay
        this.alpha = 1.0; // Start with no decay
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }

        iterator++;

        // Calls the appropriate adjustment based on stoppingCriterionType
        switch (stoppingCriterionType) {
            case Iteration:
                alpha = decayFunction.calculateAlpha(iterator, executionMaximumLimit);
                break;
            case Time:
                double elapsedSeconds = (double) (System.currentTimeMillis() - ini) / 1000;
                double timePercentage = elapsedSeconds / executionMaximumLimit;
                alpha = decayFunction.calculateAlpha((int) (timePercentage * iterator), executionMaximumLimit);
                break;
            default:
                break;
        }

        // Apply the decay factor and clamp the ideal distance
        idealDist.idealDist *= alpha;
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }

    // General method for customization of decay
    public static double computePiecewiseDecay(int currentIteration, int min, int max, double limit) {
        // Sample piecewise function based on iterations
        if (currentIteration < limit / 2) {
            return Math.pow((double) min / (double) max, (double) currentIteration / (limit / 2));
        } else {
            return 1.0; // Maintain stability in the latter half
        }
    }
}