package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator = 0;
    private long ini;
    private double executionMaximumLimit;
    private double alpha = 1;
    private StoppingCriterionType stoppingCriterionType;
    private IdealDist idealDist;

    // Enum to define decay types for enhanced flexibility
    public enum DecayType {
        LINEAR,
        EXPONENTIAL,
        COSINE,
        PIECEWISE // A new decay type that allows for customizable transitions
    }

    private DecayType decayType;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayType = DecayType.EXPONENTIAL;  // Default decay type
    }

    // Method to set the decay type ensuring it's not null
    public void setDecayType(DecayType decayType) {
        if (decayType == null) {
            throw new IllegalArgumentException("Decay type cannot be null");
        }
        this.decayType = decayType;
    }

    // Main method to adjust the diversity distance
    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }
        iterator++;

        adjustBasedOnStopCriteria(); // Determine adjustment percentage

        applyDecay(); // Apply the selected decay strategy

        // Clamping the ideal distance to maintain specified bounds
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }

    // Adjust value of alpha based on stopping criteria
    private void adjustBasedOnStopCriteria() {
        double targetRatio = 1.0 - ((double) distMMax / distMMin);
        if (stoppingCriterionType == StoppingCriterionType.Iteration) {
            alpha = calculateDecay(targetRatio);
        } else if (stoppingCriterionType == StoppingCriterionType.Time) {
            double currentTime = (double) (System.currentTimeMillis() - ini) / 1000;
            double timePercentage = currentTime / executionMaximumLimit;
            double total = (double) iterator / timePercentage;
            alpha = calculateDecay(targetRatio, total);
        }
    }

    // Apply the selected decay function based on the decay type
    private void applyDecay() {
        switch (decayType) {
            case LINEAR:
                idealDist.idealDist *= (1 - alpha); // Simple linear decay
                break;
            case EXPONENTIAL:
                idealDist.idealDist *= Math.exp(-alpha); // Exponential decay
                break;
            case COSINE:
                idealDist.idealDist *= (1 - Math.cos(Math.PI * iterator / executionMaximumLimit)); // Smooth cosine decay
                break;
            case PIECEWISE:
                idealDist.idealDist = applyPiecewiseDecay(idealDist.idealDist);
                break;
        }
    }

    // Implementation of piecewise decay for custom transitions
    private double applyPiecewiseDecay(double currentIdealDist) {
        // Example piecewise approach, you can define your segments here
        if (iterator < executionMaximumLimit / 4) {
            return currentIdealDist * 0.8; // First quarter
        } else if (iterator < executionMaximumLimit / 2) {
            return currentIdealDist * 0.6; // Second quarter
        } else {
            return currentIdealDist * 0.4; // Last half
        }
    }

    // Calculate a generic decay factor based on a decay ratio
    private double calculateDecay(double decayRatio) {
        return Math.pow(decayRatio, 1 / executionMaximumLimit);
    }

    // Overloaded version allows for more detailed control of decay calculation
    private double calculateDecay(double decayRatio, double factor) {
        return Math.pow(decayRatio, 1 / factor);
    }
}