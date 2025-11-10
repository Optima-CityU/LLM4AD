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

    private DecayFunction decayFunction;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();

        // Set a default decay function
        this.decayFunction = this::defaultDecayFunction;
    }

    public void setDecayFunction(DecayFunction decayFunction) {
        this.decayFunction = decayFunction;
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

    // Method for iterative adjustment with the selected decay function
    private double iterationAdjustment() {
        return decayFunction.calculateAlpha(iterator, executionMaximumLimit);
    }

    // Method for time-based adjustment with the selected decay function
    private double timeAdjustment() {
        double current = (double) (System.currentTimeMillis() - ini) / 1000;
        double timePercentage = current / executionMaximumLimit;
        double total = (double) iterator / timePercentage;
        return decayFunction.calculateAlpha((int) total, executionMaximumLimit); // Adapted for time
    }

    // Default decay function (piecewise function)
    private double defaultDecayFunction(int param, double limit) {
        if (param < limit) { // Early phase
            return Math.pow((double) distMMin / (double) distMMax, 1 / (limit - param + 1)); // Smoother transition
        } else { // Late phase
            return 1.0; // Maintain stability
        }
    }

    // Example of an exponential decay function
    public static DecayFunction exponentialDecay(double decayRate) {
        return (currentIteration, executionLimit) -> {
            return Math.exp(-decayRate * currentIteration / executionLimit);
        };
    }

    // Example of a cosine decay function
    public static DecayFunction cosineDecay(int maxIterations) {
        return (currentIteration, executionLimit) -> {
            return 0.5 * (1 + Math.cos(Math.PI * currentIteration / maxIterations)); // Cosine decay between 0 and 1
        };
    }

    // Additional custom decay functions can be implemented similarly
}