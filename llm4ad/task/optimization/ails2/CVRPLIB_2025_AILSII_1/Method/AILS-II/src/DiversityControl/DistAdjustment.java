package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    // Minimum and maximum diversity distances
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
        this.idealDist.idealDist = distMMax; // Start at maximum distance
        this.stoppingCriterionType = config.getStoppingCriterionType();
    }

    public void distAdjustment() {
        // Initialize iterator and starting time
        if (iterator == 0) ini = System.currentTimeMillis();
        iterator++;

        // Adjustably change distance based on the stopping criterion
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

        // Update the ideal distance with capped alpha adjustment for stability
        idealDist.idealDist *= alpha;
        idealDist.idealDist = clamp(idealDist.idealDist, distMMin, distMMax);
    }

    // Linear decay adjustment based on the iteration count
    private void iterationAdjustment() {
        alpha = getDecayFactor(distMMin, distMMax, executionMaximumLimit, iterator);
    }

    // Time-based decay adjustment
    private void timeAdjustment() {
        double current = (double) (System.currentTimeMillis() - ini) / 1000;
        double timePercentage = current / executionMaximumLimit;
        double total = (double) iterator / timePercentage;
        alpha = getDecayFactor(distMMin, distMMax, total, iterator);
    }

    // Generalized method to compute the decay factor, allowing flexibility for different decay schedules
    private double getDecayFactor(int min, int max, double decayBase, int step) {
        return Math.pow((double) min / (double) max, (1.0 / decayBase) * step);
    }

    // Clamp method to ensure the ideal distance remains within bounds
    private double clamp(double value, double min, double max) {
        return Math.min(max, Math.max(value, min));
    }
}