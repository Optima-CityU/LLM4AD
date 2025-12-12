package DiversityControl;

import java.text.DecimalFormat;
import java.util.Random;

import Auxiliary.Mean;
import Perturbation.PerturbationType;
import SearchMethod.Config;

public class OmegaAdjustment {
    private double omega, omegaMin, omegaMax;
    private Mean meanLSDist;
    private int updateCounter = 0; // Counter for the number of updates
    private final DecimalFormat decimalFormat = new DecimalFormat("0.00");
    private double lastObtainedDist;
    private Mean averageOmega;
    private double actualOmega;
    private final Random rand = new Random();
    private PerturbationType perturbationType;
    private int updateInterval;
    private IdealDist idealDist;

    // Constructor to initialize OmegaAdjustment
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
    
    // Main method to handle distance updates and possible Omega adjustments
    public void updateDistance(double distLS) {
        updateCounter++;
        meanLSDist.setValue(distLS); // Update the mean LSDist value

        // Check if it's time to update Omega
        if (shouldUpdateOmega()) {
            adjustOmega();
        }
    }
    
    // Returns the adjusted omega bound by omegaMin and omegaMax
    public double getAdjustedOmega() {
        actualOmega = Math.min(omegaMax, Math.max(omega, omegaMin));
        return actualOmega;
    }

    // Provides a formatted string representation of the current state
    @Override
    public String toString() {
        return String.format("o%s: %s | meanLSDist: %s | dMI: %s | actualOmega: %s | obtainedDist: %s | averageOmega: %s",
                perturbationType.toString().substring(4), decimalFormat.format(omega),
                meanLSDist, decimalFormat.format(idealDist.idealDist),
                decimalFormat.format(getAdjustedOmega()), lastObtainedDist,
                averageOmega);
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

    // Determines if it's time to update the omega value based on the iteration count
    private boolean shouldUpdateOmega() {
        return updateCounter % updateInterval == 0;
    }

    // Adjusts the omega value based on mean LSDist and idealDist
    private void adjustOmega() {
        lastObtainedDist = meanLSDist.getDynamicAverage();
        omega += (omega / lastObtainedDist * idealDist.idealDist) - omega;
        omega = clampOmega(omega);
        averageOmega.setValue(omega);
        updateCounter = 0; // Reset counter after the update
    }

    // Clamps the omega value within the defined bounds of omegaMin and omegaMax
    private double clampOmega(double omegaValue) {
        return Math.min(omegaMax, Math.max(omegaValue, omegaMin));
    }
}