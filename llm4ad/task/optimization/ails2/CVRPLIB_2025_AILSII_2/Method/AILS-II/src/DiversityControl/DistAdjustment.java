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
    
    // Predefined decay functions for different schedules
    public static class ExponentialDecay implements DecayFunction {
        public double calculateAlpha(int currentIteration, double executionLimit) {
            return Math.exp(-((double) currentIteration / executionLimit));
        }
    }

    public static class CosineDecay implements DecayFunction {
        public double calculateAlpha(int currentIteration, double executionLimit) {
            return 0.5 * (1 + Math.cos(Math.PI * currentIteration / executionLimit));
        }
    }

    public static class PiecewiseDecay implements DecayFunction {
        public double calculateAlpha(int currentIteration, double executionLimit) {
            if (currentIteration < executionLimit / 2) {
                return Math.pow((double) currentIteration / (executionLimit / 2), 0.5); // Squared root for early phase
            } else {
                return 1.0; // Maintain stability in late phase
            }
        }
    }

    private DecayFunction decayFunction;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit, DecayFunction decayFunction) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayFunction = decayFunction;
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }

        iterator++;
        
        // Calls the adjustment based on stoppingCriterionType
        if (stoppingCriterionType == StoppingCriterionType.Iteration) {
            alpha = decayFunction.calculateAlpha(iterator, executionMaximumLimit);
        } else if (stoppingCriterionType == StoppingCriterionType.Time) {
            double current = (double) (System.currentTimeMillis() - ini) / 1000;
            double timePercentage = current / executionMaximumLimit;
            alpha = decayFunction.calculateAlpha((int)(timePercentage * executionMaximumLimit), executionMaximumLimit);
        }
        
        // Apply the decay factor and clamp the ideal distance
        idealDist.idealDist *= alpha;
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }
}