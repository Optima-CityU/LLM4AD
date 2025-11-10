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

    // Predefined decay types
    public enum DecayType {
        LINEAR,
        EXPONENTIAL,
        COSINE
    }

    private DecayType decayType;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit, DecayType decayType) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayType = decayType; // Assign the decay type chosen by the user
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

    // Method for iterative adjustment with flexible decay
    private double iterationAdjustment() {
        return computeDecayAlpha(iterator);
    }

    // Method for time-based adjustment with flexible decay
    private double timeAdjustment() {
        double current = (double) (System.currentTimeMillis() - ini) / 1000;
        double timePercentage = current / executionMaximumLimit;
        double total = Math.max(1, (double) iterator / timePercentage); // Prevent division by zero
        return computeDecayAlpha((int) total); // Adapted for time
    }

    // General method to compute decay alpha selecting the decay type
    private double computeDecayAlpha(int param) {
        double normalizedParam = (double) param / executionMaximumLimit; // Normalizing parameter for decay function
        switch (decayType) {
            case LINEAR:
                return linearDecay(normalizedParam);
            case EXPONENTIAL:
                return exponentialDecay(normalizedParam);
            case COSINE:
                return cosineDecay(normalizedParam);
            default:
                return 1.0; // No decay
        }
    }

    // Linear decay function
    private double linearDecay(double t) {
        return Math.max((1 - t), 0); // Decays linearly to a minimum of 0
    }

    // Exponential decay function for more aggressively decreasing ideal distance
    private double exponentialDecay(double t) {
        return Math.exp(-5 * t); // Adjustable decay factor (5) for speed
    }

    // Cosine decay function for smoother transitions
    private double cosineDecay(double t) {
        return 0.5 * (1 + Math.cos(Math.PI * t)); // Ranges from 1 to 0 as t goes from 0 to 1
    }
}