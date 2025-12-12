package DiversityControl;

import java.text.DecimalFormat;
import java.util.Random;

import Auxiliary.Mean;
import Perturbation.PerturbationType;
import SearchMethod.Config;

public class OmegaAdjustment {
    private double omega, omegaMin, omegaMax;
    private Mean meanLSDist;
    private int updateCycleCount = 0;
    private final DecimalFormat decimalFormat = new DecimalFormat("0.00");
    private double obtainedDist;
    private Mean averageOmega;
    private final Random rand = new Random();
    private PerturbationType perturbationType;
    private int numIterUpdate;
    private IdealDist idealDist;

    public OmegaAdjustment(PerturbationType perturbationType, Config config, Integer size, IdealDist idealDist) {
        this.perturbationType = perturbationType;
        this.omega = idealDist.idealDist;
        this.numIterUpdate = config.getGamma();
        this.omegaMin = 1;
        this.omegaMax = size - 2;
        this.averageOmega = new Mean(numIterUpdate);
        this.meanLSDist = new Mean(numIterUpdate);
        this.idealDist = idealDist;
    }
    
    // Updates omega based on feedback from mean LSDist and idealDist
    private void updateOmega() {
        double adjustedOmega = calculateAdjustedOmega();
        storeAndClampOmega(adjustedOmega);
        resetCycleCount(); // Resetting update cycle count
    }

    // Calculate the adjusted omega based on current metrics
    private double calculateAdjustedOmega() {
        obtainedDist = meanLSDist.getDynamicAverage();
        return omega + ((omega / obtainedDist * idealDist.idealDist) - omega);
    }
    
    // Ensure omega remains within defined bounds
    private void storeAndClampOmega(double adjustedOmega) {
        omega = Math.min(omegaMax, Math.max(adjustedOmega, omegaMin));
        averageOmega.setValue(omega);
    }

    // Reset update cycle count post omega update
    private void resetCycleCount() {
        updateCycleCount = 0;
    }

    public void setDistance(double distLS) {
        updateCycleCount++;
        meanLSDist.setValue(distLS);

        // Execute omega update at specified intervals
        if (updateCycleCount % numIterUpdate == 0) {
            updateOmega();
        }
    }
    
    public double getActualOmega() {
        return Math.min(omegaMax, Math.max(omega, omegaMin));
    }

    @Override
    public String toString() {
        return String.format("o%s: %s meanLSDist%s: %s dMI%s: %s actualOmega: %s obtainedDist: %s averageOmega%s: %s",
            perturbationType.toString().substring(4),
            decimalFormat.format(omega),
            perturbationType.toString().substring(4),
            meanLSDist,
            perturbationType.toString().substring(4),
            decimalFormat.format(idealDist.idealDist),
            decimalFormat.format(getActualOmega()),
            obtainedDist,
            perturbationType.toString().substring(4),
            averageOmega);
    }

    // Getter for averageOmega
    public Mean getAverageOmega() {
        return averageOmega;
    }

    // Getter for perturbationType
    public PerturbationType getPerturbationType() {
        return perturbationType;
    }

    public void setActualOmega(double actualOmega) {
        // Considering keeping as a placeholder for future logic
        this.omega = actualOmega;
    }
}