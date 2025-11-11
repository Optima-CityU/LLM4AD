package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

/**
 * Class for dynamically adjusting the ideal diversity distance during an
 * optimization process, allowing for different decay strategies.
 */
public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long ini;
    private final double executionMaximumLimit;
    private double idealDist;
    private final StoppingCriterionType stoppingCriterionType;
    private final DecayFunction decayFunction;

    // Enum for supported decay types
    public enum DecayType {
        LINEAR,
        EXPONENTIAL,
        COSINE
    }

    // Builder for creating DistAdjustment instances
    public static class Builder {
        private final Config config;
        private final IdealDist idealDist;
        private final double executionMaximumLimit;
        private DecayFunction decayFunction;

        public Builder(IdealDist idealDist, Config config, double executionMaximumLimit) {
            this.idealDist = idealDist;
            this.config = config;
            this.executionMaximumLimit = executionMaximumLimit;
        }

        public Builder setDecayFunction(DecayFunction function) {
            this.decayFunction = function;
            return this;
        }

        public DistAdjustment build() {
            return new DistAdjustment(this);
        }
    }

    // Functional interface for decay functions
    interface DecayFunction {
        double calculate(int currentIteration, double limit);
    }

    // Constructor for DistAdjustment using Builder
    private DistAdjustment(Builder builder) {
        this.idealDist = builder.idealDist.idealDist;
        this.executionMaximumLimit = builder.executionMaximumLimit;
        this.distMMin = builder.config.getDMin();
        this.distMMax = builder.config.getDMax();
        this.stoppingCriterionType = builder.config.getStoppingCriterionType();
        this.decayFunction = builder.decayFunction != null ? builder.decayFunction : this::defaultDecay;
    }

    /**
     * Main adjustment method that updates the ideal distance.
     */
    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }

        iterator++;

        // Calls the adjustment based on stoppingCriterionType
        double adjustmentFactor = (stoppingCriterionType == StoppingCriterionType.Iteration)
                ? decayFunction.calculate(iterator, executionMaximumLimit)
                : decayFunction.calculate((int) (System.currentTimeMillis() - ini) / 1000, executionMaximumLimit);

        // Apply the decay factor and clamp ideal distance
        idealDist *= adjustmentFactor;
        idealDist = Math.min(distMMax, Math.max(idealDist, distMMin));
    }

    /**
     * Default decay function (linear).
     */
    private double defaultDecay(int current, double limit) {
        return 1 - ((double) current / limit);
    }

    /**
     * Decay function for exponential decay.
     */
    public static double exponentialDecay(int current, double limit) {
        return Math.exp(-current / limit);
    }

    /**
     * Decay function for cosine decay.
     */
    public static double cosineDecay(int current, double limit) {
        return 0.5 * (1 + Math.cos(Math.PI * current / limit));
    }
}