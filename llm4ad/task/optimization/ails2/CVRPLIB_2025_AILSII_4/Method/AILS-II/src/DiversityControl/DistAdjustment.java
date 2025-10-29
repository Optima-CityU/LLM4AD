package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long ini;
    private double executionMaximumLimit;
    private double alpha = 1;
    private StoppingCriterionType stoppingCriterionType;
    private IdealDist idealDist;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax; // Initialize to maximum diversity distance
        this.stoppingCriterionType = config.getStoppingCriterionType();
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }

        iterator++;

        // Adjust based on stopping criterion
        switch (stoppingCriterionType) {
            case Iteration:
                iterationAdjustment();
                break;
            case Time:
                timeAdjustment();
                break;
            default:
                break;
        }

        // Update idealDist with a smooth decay using alpha
        idealDist.idealDist *= alpha;
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin)); // Clamp the value
    }

    private void iterationAdjustment() {
        // Use nonlinear decay: decay faster at first, slowing down over time
        alpha = computeDecayFactor(iterator, distMMin, distMMax, executionMaximumLimit, "exponential");
    }

    private void timeAdjustment() {
        double current = (double) (System.currentTimeMillis() - ini) / 1000;
        double timePercentage = current / executionMaximumLimit;
        // Total based on time and iteration for more nuanced adjustment
        double total = (double) iterator / timePercentage;
        alpha = computeDecayFactor(total, distMMin, distMMax, executionMaximumLimit, "time-based");
    }

    private double computeDecayFactor(double input, int dMin, int dMax, double maxLimit, String method) {
        // Different decay methods can be implemented here for flexibility and adaptiveness
        switch (method) {
            case "exponential":
                return Math.pow((double) dMin / (double) dMax, (1.0 / maxLimit) * input);
            case "time-based":
                return Math.pow((double) dMin / (double) dMax, (1.0 / input));
            default:
                return 1.0; // Default to 1 if an invalid method is specified
        }
    }
}