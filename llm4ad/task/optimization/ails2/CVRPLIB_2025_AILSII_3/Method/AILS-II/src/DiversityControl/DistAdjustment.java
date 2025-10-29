package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    int distMMin;
    int distMMax;
    int iterator;
    long ini;
    double executionMaximumLimit;
    double alpha = 1;
    StoppingCriterionType stoppingCriterionType;
    IdealDist idealDist;
    DistanceDecayStrategy decayStrategy;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.decayStrategy = new ExponentialDecay(); // Default decay strategy
    }

    public void setDecayStrategy(DistanceDecayStrategy decayStrategy) {
        this.decayStrategy = decayStrategy;
    }

    public void distAdjustment() {
        if (iterator == 0) ini = System.currentTimeMillis();
        iterator++;

        adjustmentBasedOnCriterion();
        
        // Update idealDist based on the alpha value
        idealDist.idealDist *= alpha;
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }

    private void adjustmentBasedOnCriterion() {
        switch (stoppingCriterionType) {
            case Iteration:
                alpha = decayStrategy.calculateAlpha(distMMin, distMMax, iterator, executionMaximumLimit);
                break;
            case Time:
                double current = (double) (System.currentTimeMillis() - ini) / 1000;
                double timePercentage = current / executionMaximumLimit;
                alpha = decayStrategy.calculateAlpha(distMMin, distMMax, (int)(iterator / timePercentage), executionMaximumLimit);
                break;
            default:
                break;
        }
    }
}

// Strategy interface for distance decay
interface DistanceDecayStrategy {
    double calculateAlpha(int distMMin, int distMMax, int iterator, double executionMaximumLimit);
}

// Exponential decay implementation
class ExponentialDecay implements DistanceDecayStrategy {
    @Override
    public double calculateAlpha(int distMMin, int distMMax, int iterator, double executionMaximumLimit) {
        return Math.pow((double) distMMin / (double) distMMax, (double) 1 / executionMaximumLimit);
    }
}

// Add more decay strategies as needed, e.g., CosineDecay, PiecewiseDecay, etc.
class CosineDecay implements DistanceDecayStrategy {
    @Override
    public double calculateAlpha(int distMMin, int distMMax, int iterator, double executionMaximumLimit) {
        double phase = (double) iterator / executionMaximumLimit;
        return (1 + Math.cos(Math.PI * phase)) / 2; // Normalized cosine decay
    }
}

// Implement a piecewise decay strategy as needed
class PiecewiseDecay implements DistanceDecayStrategy {
    @Override
    public double calculateAlpha(int distMMin, int distMMax, int iterator, double executionMaximumLimit) {
        // Custom piecewise decay logic based on certain criteria
        // Placeholder for illustration
        if (iterator < executionMaximumLimit / 2) {
            return Math.pow(((double) distMMin / distMMax), (double) 1 / (executionMaximumLimit / 2));
        } else {
            return 1.0; // No decay in the latter half
        }
    }
}