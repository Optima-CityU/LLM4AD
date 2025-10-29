package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long ini;
    private final double executionMaximumLimit;
    private final StoppingCriterionType stoppingCriterionType;
    private final IdealDist idealDist;

    // Functional interface for custom decay strategies
    @FunctionalInterface
    public interface DecayFunction {
        double calculateDecay(int currentIteration, double executionLimit);
    }

    private DecayFunction decayFunction; // Holds the custom decay function

    // Default decay function - Exponential decay
    private static final DecayFunction DEFAULT_DECAY_FUNCTION = (currentIteration, executionLimit) -> {
        return Math.exp(-((double) currentIteration / executionLimit));
    };

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayFunction = DEFAULT_DECAY_FUNCTION; // Initialize with default
    }

    // Method to set a custom decay function
    public void setCustomDecayFunction(DecayFunction decayFunction) {
        if (decayFunction == null) {
            throw new IllegalArgumentException("Decay function cannot be null");
        }
        this.decayFunction = decayFunction; // Set custom decay function
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis(); // Initialize time tracking
        }

        iterator++;

        // Calculate decay based on stopping criteria
        double decayFactor = calculateDecayFactor();
        
        // Apply the decay to the ideal distance
        idealDist.idealDist *= decayFactor;
        // Clamp the ideal distance to defined limits
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }

    // Calculate the decay factor based on the stopping criteria
    private double calculateDecayFactor() {
        switch (stoppingCriterionType) {
            case Iteration:
                return decayFunction.calculateDecay(iterator, executionMaximumLimit);
            case Time:
                double elapsed = (double) (System.currentTimeMillis() - ini) / 1000;
                // Normalizing elapsed time with the maximum limit
                return decayFunction.calculateDecay(iterator, elapsed / executionMaximumLimit);
            default:
                return 1.0; // Default to no change if no valid criterion
        }
    }
}