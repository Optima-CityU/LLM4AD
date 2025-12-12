```java
package DiversityControl;

import java.text.DecimalFormat;
import java.util.Random;

import Auxiliary.Mean;
import Perturbation.PerturbationType;
import SearchMethod.Config;

public class OmegaAdjustment {
    private double omega, omegaMin, omegaMax;
    private Mean meanLSDist;
    private int updateCounter = 0;
    private final DecimalFormat decimalFormat = new DecimalFormat("0.00");
    private double obtainedDist;
    
    private Mean averageOmega;
    private double actualOmega;
    private final Random randomGenerator = new Random();
    private PerturbationType perturbationType;
    private int updateInterval; // Number of iterations before omega is updated
    private IdealDist idealDist;

    public OmegaAdjustment(PerturbationType perturbationType, Config config, Integer size, IdealDist idealDist) {
        this.perturbationType = perturbationType;
        this.omega = idealDist.idealDist;
        this.updateInterval = config.getGamma();
        this.omegaMin = 1;
        this.omegaMax = size - 2;
        this.averageOmega = new Mean(updateInterval);
        this.meanLSDist = new Mean(updateInterval);
        this.idealDist = idealDist;
    }
    
    // Main method to update omega based on collected distances
    private void updateOmega() {
        double adjustedOmega = calculateAdjustedOmega();
        omega = clampOmega(adjustedOmega);
        averageOmega.setValue(omega);
        resetUpdateCounter();
    }
    
    // Calculate the new adjusted omega based on the mean LSDist
    private double calculateAdjustedOmega() {
        obtainedDist = meanLSDist.getDynamicAverage();
        return omega + ((omega / obtainedDist * idealDist.idealDist) - omega);
    }

    // Ensure omega is within defined bounds
    private double clampOmega(double value) {
        return Math.min(omegaMax, Math.max(value, omegaMin));
    }

    // Increment the counter for distance updates and check for omega update
    public void setDistance(double distLS) {
        updateCounter++;
        meanLSDist.setValue(distLS);

        if (isTimeToUpdateOmega()) {
            updateOmega();
        }
    }

    // Determine if it’s time to update omega based on the configured interval
    private boolean isTimeToUpdateOmega() {
        return updateCounter % updateInterval == 0;
    }

    // Get the current valid omega value
    public double getActualOmega() {
        actualOmega = clampOmega(this.omega); // Ensure actualOmega is within bounds
        return actualOmega;
    }

    @Override
    public String toString() {
        return String.format("o%s: %s meanLSDist%s: %s dMI%s: %s actualOmega: %s obtainedDist: %s averageOmega%s: %s",
            perturbationType.toString().substring(4), decimalFormat.format(omega),
            perturbationType.toString().substring(4), meanLSDist.getDynamicAverage(),
            perturbationType.toString().substring(4), decimalFormat.format(idealDist.idealDist),
            decimalFormat.format(getActualOmega()), obtainedDist,
            perturbationType.toString().substring(4), averageOmega);
    }

    public Mean getAverageOmega() {
        return averageOmega;
    }

    public PerturbationType getPerturbationType() {
        return perturbationType;
    }

    public void setActualOmega(double actualOmega) {
        this.actualOmega = actualOmega;
    }

    private void resetUpdateCounter() {
        updateCounter = 0; // Reset the counter after updating omega
    }
}
```