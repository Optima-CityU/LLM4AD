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
    
    private double lastObtainedDistance;
    private Mean averageOmega;

    private double actualOmega;
    private final Random random;
    private PerturbationType perturbationType;
    private int iterationsUntilUpdate;
    private IdealDist idealDistance;

    public OmegaAdjustment(PerturbationType perturbationType, Config config, Integer size, IdealDist idealDistance) {
        this.perturbationType = perturbationType;
        this.idealDistance = idealDistance;
        this.random = new Random();
        this.omega = idealDistance.idealDist;
        this.iterationsUntilUpdate = config.getGamma();
        this.omegaMin = 1;
        this.omegaMax = size - 2;
        this.averageOmega = new Mean(iterationsUntilUpdate);
        this.meanLSDist = new Mean(iterationsUntilUpdate);
    }

    // Public method to set distance and potentially update omega
    public void setDistance(double distance) {
        updateCounter++;
        meanLSDist.setValue(distance);
        if (shouldUpdateOmega()) {
            updateOmega();
        }
    }

    // Check if it's time to update omega
    private boolean shouldUpdateOmega() {
        return updateCounter % iterationsUntilUpdate == 0;
    }

    // Core logic to update omega based on mean distance and ideal distance
    private void updateOmega() {
        lastObtainedDistance = meanLSDist.getDynamicAverage();
        double adjustedOmega = calculateNewOmega();
        omega = constrainOmega(adjustedOmega);
        averageOmega.setValue(omega);
        resetUpdateCounter();
    }

    // Calculate new omega value based on current parameters
    private double calculateNewOmega() {
        return omega + ((omega / lastObtainedDistance * idealDistance.idealDist) - omega);
    }

    // Constrain omega within specified limits
    private double constrainOmega(double value) {
        return Math.min(omegaMax, Math.max(value, omegaMin));
    }

    // Reset the update counter after updating omega
    private void resetUpdateCounter() {
        updateCounter = 0;
    }

    // Retrieve the constrained actual omega value
    public double getActualOmega() {
        actualOmega = constrainOmega(omega);
        return actualOmega;
    }

    // String representation of the current state for debugging and logging
    @Override
    public String toString() {
        return String.format("OmegaAdjustment [Type: %s, Omega: %s, Mean LSDist: %s, Ideal Dist: %s, Actual Omega: %s, Last Distance: %s, Average Omega: %s]",
                perturbationType, decimalFormat.format(omega), meanLSDist, decimalFormat.format(idealDistance.idealDist), 
                decimalFormat.format(getActualOmega()), lastObtainedDistance, averageOmega);
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
}