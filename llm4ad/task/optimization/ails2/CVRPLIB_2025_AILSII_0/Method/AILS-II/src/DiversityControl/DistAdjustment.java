package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

/**
 * This class dynamically adjusts the ideal diversity distance
 * during optimization processes based on iteration count or execution time.
 */
public class DistAdjustment {
    private final int distMMin;
    private final int distMMax;
    private int iterator;
    private long ini;
    private final double executionMaximumLimit;
    private double idealDist;

    private final StoppingCriterionType stoppingCriterionType;

    // Interface for custom decay functions
    interface DecayFunction {
        double calculateAlpha(int currentIteration, double executionLimit);
    }

    private DecayFunction decayFunction;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit, DecayFunction decayFunction) {
        this.idealDist = idealDist.idealDist = config.getDMax();
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayFunction = decayFunction;
        this.ini = System.currentTimeMillis();
    }

    /** 
     * Adjusts the ideal distance based on the stopping criterion 
     */
    public void distAdjustment() {
        iterator++;

        // Calculate alpha using the provided decay function
        double alpha = (stoppingCriterionType == StoppingCriterionType.Iteration) 
                       ? decayFunction.calculateAlpha(iterator, executionMaximumLimit)
                       : decayFunction.calculateAlpha(getElapsedTime(), executionMaximumLimit);

        // Apply the decay factor and clamp the ideal distance
        idealDist.idealDist = Math.min(distMMax, Math.max(distMMin, idealDist.idealDist * alpha));
    }

    // Get the elapsed time in seconds
    private double getElapsedTime() {
        return (double) (System.currentTimeMillis() - ini) / 1000;
    }

    /** Example decay function implementation: Exponential decay */
    public static DecayFunction exponentialDecay(double decayRate) {
        return (currentIteration, executionLimit) -> 
            Math.exp(-decayRate * (currentIteration / executionLimit));
    }

    /** Example decay function implementation: Piecewise linear decay */
    public static DecayFunction piecewiseDecay(double switchPoint) {
        return (currentIteration, executionLimit) -> {
            if (currentIteration < switchPoint) {
                return 1 - (currentIteration / switchPoint);
            } else {
                return 0.5; // Maintain a baseline post switch
            }
        };
    }

    /** Example decay function implementation: Cosine decay */
    public static DecayFunction cosineDecay() {
        return (currentIteration, executionLimit) -> 
            0.5 * (1 + Math.cos(Math.PI * currentIteration / executionLimit));
    }
}