package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long ini;
    private final double executionMaximumLimit;
    private double idealDist;
    private final StoppingCriterionType stoppingCriterionType;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.executionMaximumLimit = executionMaximumLimit;
        this.stoppingCriterionType = config.getStoppingCriterionType();
        this.idealDist = distMMax; // Start with max ideal distance
        idealDist.idealDist = this.idealDist;
    }

    public void distAdjustment() {
        if (iterator == 0) {
            ini = System.currentTimeMillis();
        }

        iterator++;

        // Determine the adjustment factor based on the stopping criteria
        double alpha = calculateAdjustmentFactor();
        
        // Update the ideal distance and clamp it within the specified bounds
        idealDist *= alpha;
        idealDist = Math.min(distMMax, Math.max(idealDist, distMMin));
        idealDist.idealDist = idealDist; // Update the idealDist reference
    }

    // Calculate the alpha adjustment factor using a sigmoid-based decay approach
    private double calculateAdjustmentFactor() {
        double phaseTime = (System.currentTimeMillis() - ini) / 1000.0; // Convert to seconds
        double normalizedTime = Math.min(phaseTime / executionMaximumLimit, 1.0); // Clamp to [0, 1]
        
        // Sigmoid decay function for the adjustment factor
        double decayFactor = 1 / (1 + Math.exp(-12 * (normalizedTime - 0.5))); // Steep sigmoid transition
        
        return distMMin + (distMMax - distMMin) * decayFactor; // Return the adjusted alpha
    }
}