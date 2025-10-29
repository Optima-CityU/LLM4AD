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

    // DecayFunction interface for customizable decay schedules
    interface DecayFunction {
        double calculateAlpha(int currentIteration, double executionLimit);
    }

    // SigmoidDecayFunction implements a smooth transition approach
    static class SigmoidDecayFunction implements DecayFunction {
        private final double steepness;
        private final double midpoint;

        public SigmoidDecayFunction(double steepness, double midpoint) {
            this.steepness = steepness;
            this.midpoint = midpoint;
        }

        @Override
        public double calculateAlpha(int currentIteration, double executionLimit) {
            // Utilizing the sigmoid function for smooth gradual adjustment
            return 1 / (1 + Math.exp(-steepness * (currentIteration - midpoint)));
        }
    }

    // Default decay function instance
    private final DecayFunction decayFunction;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayFunction = new SigmoidDecayFunction(0.1, executionMaximumLimit / 2);  // Example settings
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }

        iterator++;

        // Calculate decay factor
        double alpha = decayFunction.calculateAlpha(iterator, executionMaximumLimit);

        // Apply the decay factor
        idealDist.idealDist *= alpha;

        // Ensure idealDist is clamped within defined limits
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }
}