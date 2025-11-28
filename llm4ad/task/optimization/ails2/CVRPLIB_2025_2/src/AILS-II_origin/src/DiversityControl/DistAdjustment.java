package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long startTime;
    private final double executionMaximumLimit;
    private double idealDistance;
    private final StoppingCriterionType stoppingCriterionType;

    // DecayFunction interface for adaptive decay schedules
    interface DecayFunction {
        double calculateAlpha(int currentIteration, double executionLimit);
    }

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDistance = distMMax; // Start with max ideal distance
        this.executionMaximumLimit = executionMaximumLimit;
        this.stoppingCriterionType = config.getStoppingCriterionType();
    }

    public void distAdjustment() {
        // Initialize the starting time if at the beginning
        if (iterator == 0) {
            startTime = System.currentTimeMillis();
        }

        iterator++;

        // Adjust the alpha based on stopping criterion type
        double alpha = (stoppingCriterionType == StoppingCriterionType.Iteration) ?
                iterationAdjustment() : timeAdjustment();

        // Update the ideal distance with clamping to the min and max
        idealDistance *= alpha;
        idealDistance = Math.min(distMMax, Math.max(idealDistance, distMMin));
    }

    // Method for iterative adjustment with nonlinear decay
    private double iterationAdjustment() {
        return computeDecayAlpha(iterator, executionMaximumLimit);
    }

    // Method for time-based adjustment with nonlinear decay
    private double timeAdjustment() {
        double elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0;
        double progressRatio = elapsedTime / executionMaximumLimit;
        double adjustedIteration = iterator * Math.pow(progressRatio, 2); // Exponential decay with iteration
        return computeDecayAlpha(adjustedIteration, executionMaximumLimit);
    }

    // General method to compute decay alpha for flexibility using a nonlinear function
    private double computeDecayAlpha(double currentPhase, double limit) {
        // Using a sigmoid-like decay for a smoother transition (ranges between 0 and 1)
        double normalizedPhase = Math.min(currentPhase / limit, 1); 
        return 1.0 / (1.0 + Math.exp((1.0 - normalizedPhase) * 10.0 - 5.0)); // Sigmoid curve for alpha
    }

    public double getIdealDistance() {
        return idealDistance; // Added getter for ideal distance for potential external access
    }
}