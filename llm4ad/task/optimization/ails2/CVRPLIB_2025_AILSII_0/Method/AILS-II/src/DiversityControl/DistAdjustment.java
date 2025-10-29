package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment {
    private int distMMin;                      // Minimum diversity distance
    private int distMMax;                      // Maximum diversity distance
    private int iterator;                       // Iteration count
    private long ini;                           // Initial time for calculation
    private double executionMaximumLimit;       // Maximum execution limit
    private double alpha = 1;                   // Decay factor
    private StoppingCriterionType stoppingCriterionType; // Type of stopping criterion
    private IdealDist idealDist;                // Ideal distance object

    // Constructor initializing member variables
    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();
    }

    // Method to adjust diversity distance
    public void distAdjustment() {
        if (iterator == 0) ini = System.currentTimeMillis(); // Initialize time tracking
        iterator++;
        
        // Adjust based on stopping criterion
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
        
        // Update ideal distance using decay factor while ensuring stability
        idealDist.idealDist *= alpha;
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }

    // Method to adjust decay based on the number of iterations
    private void iterationAdjustment() {
        // Nonlinear decay using a smooth exponential function
        alpha = computeDecayFactor(iterator, distMMin, distMMax, executionMaximumLimit);
    }

    // Method to adjust decay based on elapsed time
    private void timeAdjustment() {
        double current = (double)(System.currentTimeMillis() - ini) / 1000; // Current time in seconds
        double timePercentage = current / executionMaximumLimit;
        // Ensure to calculate decay factor based on time elapsed
        alpha = computeDecayFactor((int)(timePercentage * executionMaximumLimit), distMMin, distMMax, executionMaximumLimit);
    }
    
    // A method that computes a flexible decay factor based on given parameters
    private double computeDecayFactor(int currentIterator, int dMin, int dMax, double execLimit) {
        if (currentIterator <= 0) return 1; // Early return for stability
        // Compute a decay based on exponential mapping for flexibility and nonlinear approaches
        return Math.pow((double) dMin / (double) dMax, (double) 1 / execLimit) * 
               Math.sin(Math.PI * (currentIterator / execLimit)); // Example of cosine adjustment for nonlinearity
    }
}