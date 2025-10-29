package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment 
{
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long ini;
    private double executionMaximumLimit;
    private double alpha = 1;
    private StoppingCriterionType stoppingCriterionType;
    private IdealDist idealDist;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) 
    {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax; // Start at maximum diversity distance
        this.stoppingCriterionType = config.getStoppingCriterionType();
    }

    public void distAdjustment() 
    {
        if (iterator == 0)
            ini = System.currentTimeMillis();

        iterator++;
        
        // Adjust distance based on configured stopping criterion
        switch (stoppingCriterionType) 
        {
            case Iteration: 
                iterationAdjustment(); 
                break;
            case Time: 
                timeAdjustment(); 
                break;
            default:
                break;       
        }
        
        // Apply the adjustment factor and clamp the ideal distance within boundaries
        idealDist.idealDist *= alpha;
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }
    
    private void iterationAdjustment() 
    {
        alpha = computeDecayFactor(distMMin, distMMax, executionMaximumLimit, iterator);
    }
    
    private void timeAdjustment() 
    {
        double current = (double)(System.currentTimeMillis() - ini) / 1000; // Convert from milliseconds to seconds
        double timePercentage = current / executionMaximumLimit;
        double total = (double)iterator / timePercentage;
        alpha = computeDecayFactor(distMMin, distMMax, total, iterator);
    }

    private double computeDecayFactor(int min, int max, double total, int iter) 
    {
        // Here we use an exponential decay strategy to adjust alpha more smoothly
        // This can be adapted to different decay schedules as needed
        double normalizedDecay = (double)(min) / (double)(max);
        return Math.pow(normalizedDecay, 1.0 / Math.max(total, 1)); // Prevent division by zero
    }

    // Optional: You could create additional methods for other decay functions like cosine or piecewise if required
}