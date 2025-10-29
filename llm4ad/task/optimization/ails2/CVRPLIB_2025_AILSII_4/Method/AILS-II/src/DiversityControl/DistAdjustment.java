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

    // Enum to represent different types of decay schedules
    public enum DecayType {
        LINEAR,
        EXPONENTIAL,
        COSINE,
        LOGARITHMIC,
        PIECEWISE
    }

    private DecayType decayType;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax; // Start with maximum ideal distance
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayType = DecayType.EXPONENTIAL; // Default decay method
    }

    // Public method to set the decay type
    public void setDecayType(DecayType decayType) {
        this.decayType = decayType;
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis(); // Initialize the start time
        }

        iterator++;

        // Adjust alpha based on the selected stopping criterion
        switch (stoppingCriterionType) {
            case Iteration:
                iterationAdjustment();
                break;
            case Time:
                timeAdjustment();
                break;
            default:
                break;
        }

        // Apply decay calculation based on selected decay type
        applyDecay();
        
        // Ensure idealDist remains within defined limits
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }

    // Adjustment of alpha based on the number of iterations
    private void iterationAdjustment() {
        double decayRatio = (double) (iterator < distMMin ? distMMin : distMMax) / distMMax;
        alpha = calculateDecay(decayRatio);
    }

    // Adjustment of alpha based on the elapsed time
    private void timeAdjustment() {
        double currentTime = (double) (System.currentTimeMillis() - ini) / 1000;
        double timePercentage = currentTime / executionMaximumLimit;
        double total = (double) iterator / Math.max(timePercentage, 1);
        alpha = calculateDecay(1.0 - (double) distMMax / distMMin, total);
    }

    // Apply decay calculation based on selected decay type
    private void applyDecay() {
        switch (decayType) {
            case LINEAR:
                idealDist.idealDist *= alpha;
                break;
            case EXPONENTIAL:
                idealDist.idealDist *= Math.exp(-alpha);
                break;
            case COSINE:
                idealDist.idealDist *= (1 - Math.cos(Math.PI * ((double) iterator / executionMaximumLimit)));
                break;
            case LOGARITHMIC:
                idealDist.idealDist *= (1 / (1 + Math.log(1 + iterator)));
                break;
            case PIECEWISE:
                idealDist.idealDist *= piecewiseDecay(iterator);
                break;
        }
    }

    // Calculate the decay factor with better stability
    private double calculateDecay(double decayRatio) {
        return Math.pow(decayRatio, 1.0 / Math.max(executionMaximumLimit, 1));
    }

    // Overloaded method for calculating decay with additional parameters
    private double calculateDecay(double decayRatio, double factor) {
        return Math.pow(decayRatio, 1.0 / Math.max(factor, 1));
    }

    // Implementing a simple piecewise function as an example
    private double piecewiseDecay(int iterator) {
        if (iterator < executionMaximumLimit / 2) {
            return 1.0 - (double) iterator / (executionMaximumLimit / 2);
        } else {
            return 0.5; // Constant decay after half the iterations
        }
    }
}