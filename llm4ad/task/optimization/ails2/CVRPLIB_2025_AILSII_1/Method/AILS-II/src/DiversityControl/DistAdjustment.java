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
    private final DecayFunction decayFunction; // New field for customizable decay function

    // DecayFunction interface for adaptive decay schedules
    interface DecayFunction {
        double calculateAlpha(int currentIteration, double executionLimit);
    }

    // Constructor for linear decay
    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this(idealDist, config, executionMaximumLimit, DistAdjustment::linearDecay);
    }

    // Constructor allowing custom decay function
    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit, DecayFunction decayFunction) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayFunction = decayFunction; // Set the custom decay function
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

    // Method for iterative adjustment considering decay function
    private double iterationAdjustment() {
        return decayFunction.calculateAlpha(iterator, executionMaximumLimit);
    }

    // Method for time-based adjustment considering decay function
    private double timeAdjustment() {
        double current = (double) (System.currentTimeMillis() - ini) / 1000;
        double timePercentage = current / executionMaximumLimit;
        return decayFunction.calculateAlpha((int) (timePercentage * iterator), executionMaximumLimit); // Adapted for time
    }

    // Example linear decay function
    private static double linearDecay(int currentIteration, double limit) {
        return Math.max(1.0 - (double) currentIteration / limit, 0.1); // Example linear approach to decay
    }

    // Example exponential decay function
    private static double exponentialDecay(int currentIteration, double limit) {
        return Math.exp(-((double) currentIteration / limit)); // Exponential decay function
    }

    // Example cosine decay function
    private static double cosineDecay(int currentIteration, double limit) {
        return (1 + Math.cos(Math.PI * currentIteration / limit)) / 2; // Cosine decay function
    }
}