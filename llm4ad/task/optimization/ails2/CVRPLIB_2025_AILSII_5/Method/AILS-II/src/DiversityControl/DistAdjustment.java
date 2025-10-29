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

    // DecayFunction interface for adaptive decay schedules
    interface DecayFunction {
        double calculateAlpha(int currentIteration, double executionLimit);
    }

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
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

    // Method for iterative adjustment with flexible cosine decay
    private double iterationAdjustment() {
        return computeDecayAlpha(distMMin, distMMax, executionMaximumLimit, iterator);
    }

    // Method for time-based adjustment with flexible cosine decay
    private double timeAdjustment() {
        double current = (double) (System.currentTimeMillis() - ini) / 1000;
        double timePercentage = current / executionMaximumLimit;
        double total = (double) iterator / timePercentage;
        return computeDecayAlpha(distMMin, distMMax, total, 1); // Adapted for time
    }

    // General method to compute decay alpha for flexibility using cosine decay
    private double computeDecayAlpha(int dMin, int dMax, double limit, double param) {
        double normalizedParam = param / limit;
        normalizedParam = Math.max(0.0, Math.min(1.0, normalizedParam)); // Clamping to [0, 1]

        // Cosine decay function for smooth transitions
        return 0.5 * (1 + Math.cos(Math.PI * normalizedParam)); // Outputs [0, 1]
    }
}