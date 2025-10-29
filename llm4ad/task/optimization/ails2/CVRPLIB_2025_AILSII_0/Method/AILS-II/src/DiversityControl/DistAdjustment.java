package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long ini;
    private double executionMaximumLimit;
    private double alpha = 1;
    private StoppingCriterionType stoppingCriterionType;
    private IdealDist idealDist;
    
    // Interface for customizable decay functions
    public interface DecayFunction {
        double calculate(double currentDist, double alpha, int iteration, double maxIteration);
    }

    // Default decay functions
    public static class DecayFunctions {
        public static double linearDecay(double currentDist, double alpha, int iteration, double maxIteration) {
            return currentDist * alpha;
        }

        public static double exponentialDecay(double currentDist, double alpha, int iteration, double maxIteration) {
            return currentDist * Math.exp(-alpha);
        }

        public static double cosineDecay(double currentDist, double alpha, int iteration, double maxIteration) {
            return currentDist * (1 - Math.cos(Math.PI * (iteration / maxIteration)));
        }
        
        public static double sophisticatedDecay(double currentDist, double alpha, int iteration, double maxIteration) {
            double decayFactor = 1 / (1 + Math.exp(-alpha * (iteration - (maxIteration / 2))));
            return currentDist * decayFactor;
        }
    }

    private DecayFunction decayFunction;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayFunction = DecayFunctions.exponentialDecay;  // Default decay function
    }

    // Public method to set custom decay function
    public void setDecayFunction(DecayFunction decayFunction) {
        if (decayFunction == null) {
            throw new IllegalArgumentException("Decay function cannot be null");
        }
        this.decayFunction = decayFunction;
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }

        iterator++;

        // Adjust alpha based on stopping criteria
        adjustBasedOnStopCriteria();

        // Apply decay calculation using the custom decay function
        applyDecay();

        // Ensure idealDist remains within defined limits
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }

    // Adjust based on stopping criteria
    private void adjustBasedOnStopCriteria() {
        if (stoppingCriterionType == StoppingCriterionType.Iteration) {
            alpha = calculateDecay(1.0 - (double) distMMax / distMMin);
        } else if (stoppingCriterionType == StoppingCriterionType.Time) {
            double current = (double) (System.currentTimeMillis() - ini) / 1000;
            double timePercentage = current / executionMaximumLimit;
            double total = (double) iterator / timePercentage;
            alpha = calculateDecay(1.0 - (double) distMMax / distMMin, total);
        }
    }

    // Apply the selected decay function
    private void applyDecay() {
        idealDist.idealDist = decayFunction.calculate(idealDist.idealDist, alpha, iterator, executionMaximumLimit);
    }

    // Calculate decay based on a linear decay ratio
    private double calculateDecay(double decayRatio) {
        return Math.pow(decayRatio, 1 / executionMaximumLimit);
    }

    // Overloaded method for decay calculation with additional parameters for flexibility
    private double calculateDecay(double decayRatio, double factor) {
        return Math.pow(decayRatio, 1 / factor);
    }
}