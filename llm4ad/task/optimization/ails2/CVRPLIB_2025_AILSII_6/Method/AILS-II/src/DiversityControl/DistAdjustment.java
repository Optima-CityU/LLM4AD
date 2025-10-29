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

    // Enum for decay types to provide flexibility
    public enum DecayType {
        LINEAR,
        EXPONENTIAL,
        COSINE,
        PIECEWISE
    }

    // Selected decay function
    private DecayType decayType;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit, DecayType decayType) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayType = decayType;
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }

        iterator++;

        // Calls the appropriate adjustment based on stoppingCriterionType
        alpha = (stoppingCriterionType == StoppingCriterionType.Iteration) 
            ? iterationAdjustment() 
            : timeAdjustment();

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
        return computeDecayAlpha((int) (timePercentage * iterator)); // adapt to time
    }

    // General method to compute decay alpha for flexibility
    private double computeDecayAlpha(int param) {
        double normalizedParam = (double) param / (double) (distMMax / 2); // Normalizing for decay adjustments.
        
        switch (decayType) {
            case LINEAR:
                return Math.max(0, 1 - normalizedParam); // Linear decay

            case EXPONENTIAL:
                return Math.exp(-normalizedParam); // Exponential decay

            case COSINE:
                return 0.5 * (1 + Math.cos(Math.PI * normalizedParam)); // Cosine decay

            case PIECEWISE:
                return piecewiseDecay(normalizedParam); // Piecewise decay function

            default:
                return 1.0; // Default maintain stability
        }
    }

    // Example piecewise decay function
    private double piecewiseDecay(double normalizedParam) {
        if (normalizedParam < 0.5) {
            return 2 * normalizedParam; // Increasing phase
        } else {
            return 2 * (1 - normalizedParam); // Decreasing phase
        }
    }
}