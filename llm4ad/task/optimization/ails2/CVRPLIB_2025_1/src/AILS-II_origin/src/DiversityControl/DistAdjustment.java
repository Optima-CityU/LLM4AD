package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;

public class DistAdjustment {
    private int distMMin;
    private int distMMax;
    private int iterator;
    private long startTime;
    private final double executionMaximumLimit;
    private double alpha = 1;
    private final StoppingCriterionType stoppingCriterionType;
    private final IdealDist idealDist;

    // Mapping of decay functions for flexibility
    private final Map<StoppingCriterionType, BiFunction<Integer, Double, Double>> decayFunctions;

    public DistAdjustment(IdealDist idealDist, Config config, double executionMaximumLimit) {
        this.idealDist = idealDist;
        this.executionMaximumLimit = executionMaximumLimit;
        this.distMMin = config.getDMin();
        this.distMMax = config.getDMax();
        this.idealDist.idealDist = distMMax;
        this.stoppingCriterionType = config.getStoppingCriterionType();

        // Initializing the decay functions
        decayFunctions = new HashMap<>();
        decayFunctions.put(StoppingCriterionType.Iteration, this::iterationDecay);
        decayFunctions.put(StoppingCriterionType.Time, this::timeDecay);
    }

    public void distAdjustment() {
        if (iterator == 0) {
            startTime = System.currentTimeMillis();
        }

        iterator++;

        // Get the corresponding decay function
        BiFunction<Integer, Double, Double> decayFunction = decayFunctions.get(stoppingCriterionType);
        if (decayFunction != null) {
            alpha = decayFunction.apply(iterator, executionMaximumLimit);
        }

        // Apply the decay factor and clamp the ideal distance
        idealDist.idealDist *= alpha;
        idealDist.idealDist = Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
    }

    // Iterative decay logic
    private double iterationDecay(int currentIteration, double limit) {
        return computeDecayAlpha(distMMin, distMMax, limit, currentIteration);
    }

    // Time-based decay logic
    private double timeDecay(int iteration, double limit) {
        double elapsedTime = (double) (System.currentTimeMillis() - startTime) / 1000;
        double timePercentage = elapsedTime / limit;
        return computeDecayAlpha(distMMin, distMMax, 1, (int) Math.ceil(iteration / timePercentage)); // Normalizing iteration
    }

    // General decay computation for flexibility
    private double computeDecayAlpha(int dMin, int dMax, double limit, double param) {
        // The function can be updated to support various forms (linear, exponential, etc.)
        if (param < limit) { // Early phase
            // Smooth transition using exponential decay
            return Math.pow((double) dMin / (double) dMax, 1 / (limit - param + 1));
        } else { // Late phase
            return 1.0; // Maintain stability
        }
    }
}