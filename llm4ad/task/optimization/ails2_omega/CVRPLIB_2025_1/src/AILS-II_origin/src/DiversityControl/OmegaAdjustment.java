package DiversityControl;

import java.text.DecimalFormat;
import java.util.Random;

import Auxiliary.Mean;
import Perturbation.PerturbationType;
import SearchMethod.Config;

public class OmegaAdjustment {
    private double omega, omegaMin, omegaMax;
    private Mean meanLSDist;
    private int iterationCount = 0;
    private final DecimalFormat decimalFormat = new DecimalFormat("0.00");
    private double obtainedDist;
    private Mean averageOmega;
    private double actualOmega;
    private final Random rand = new Random();
    private PerturbationType perturbationType;
    private int numIterUpdate;
    private IdealDist idealDist;

    // Constructor initializes the OmegaAdjustment with necessary parameters.
    public OmegaAdjustment(PerturbationType perturbationType, Config config, Integer size, IdealDist idealDist) {
        this.perturbationType = perturbationType;
        this.omega = idealDist.idealDist; // Initialize omega with ideal distance.
        this.numIterUpdate = config.getGamma();
        this.omegaMin = 1; // Set minimum omega limit.
        this.omegaMax = size - 2; // Set maximum omega limit depending on problem size.
        this.averageOmega = new Mean(numIterUpdate);
        this.meanLSDist = new Mean(numIterUpdate);
        this.idealDist = idealDist;
    }
    
    // Updates omega based on mean LSDist and idealDist using a scaled adjustment.
    private void updateOmega() {
        obtainedDist = meanLSDist.getDynamicAverage(); // Get average distance from meanLSDist.
        omega += ((omega / obtainedDist * idealDist.idealDist) - omega); // Adjust omega.
        omega = Math.min(omegaMax, Math.max(omega, omegaMin)); // Ensure 'omega' stays within limits.
        averageOmega.setValue(omega); // Update the average omega value.
        iterationCount = 0; // Reset iteration count after update.
    }
    
    // Receives distance value and triggers omega update when needed.
    public void setDistance(double distLS) {
        iterationCount++;
        meanLSDist.setValue(distLS); // Update mean LSDist with new distance.

        // Update omega at specified intervals.
        if (iterationCount % numIterUpdate == 0) {
            updateOmega();
        }
    }
    
    // Returns the current value of omega clamped within the specified limits.
    public double getActualOmega() {
        actualOmega = Math.min(omegaMax, Math.max(omega, omegaMin));
        return actualOmega;
    }

    // String representation of OmegaAdjustment for easy debugging and logging.
    @Override
    public String toString() {
        return String.format("PerturbationType: %s, Current omega: %s, Mean LSDist: %s, Ideal Dist: %s, Actual Omega: %s, Obtained Dist: %s, Average Omega: %s",
            perturbationType.toString().substring(4),
            decimalFormat.format(omega),
            meanLSDist,
            decimalFormat.format(idealDist.idealDist),
            decimalFormat.format(getActualOmega()),
            obtainedDist,
            averageOmega);
    }

    // Getter for average omega.
    public Mean getAverageOmega() {
        return averageOmega;
    }

    // Getter for perturbation type.
    public PerturbationType getPerturbationType() {
        return perturbationType;
    }

    // Setter for actual omega, if necessary for external adjustments.
    public void setActualOmega(double actualOmega) {
        this.actualOmega = actualOmega;
    }
}