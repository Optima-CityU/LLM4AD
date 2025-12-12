package DiversityControl;

import java.text.DecimalFormat;
import java.util.Random;

import Auxiliary.Mean;
import Perturbation.PerturbationType;
import SearchMethod.Config;

public class OmegaAdjustment {
    private double omega, omegaMin, omegaMax;
    private Mean meanLSDist;
    private int iterator = 0;
    private final DecimalFormat decimalFormat = new DecimalFormat("0.00");
    private double obtainedDist;

    private Mean averageOmega;
    private double actualOmega;
    private final Random rand = new Random();
    private PerturbationType perturbationType;
    private int numIterUpdate;
    private IdealDist idealDist;

    public OmegaAdjustment(PerturbationType perturbationType, Config config, Integer size, IdealDist idealDist) {
        this.perturbationType = perturbationType;
        this.omega = idealDist.idealDist;
        this.numIterUpdate = config.getGamma();
        this.omegaMin = 1; // Minimum bound for omega
        this.omegaMax = size - 2; // Maximum bound for omega
        this.averageOmega = new Mean(numIterUpdate);
        this.meanLSDist = new Mean(numIterUpdate);
        this.idealDist = idealDist;
    }

    // Updates omega based on the average distance from the mean LSDist and the idealDist
    private void updateOmega() {
        obtainedDist = meanLSDist.getDynamicAverage();
        omega = calculateNewOmega(obtainedDist);
        averageOmega.setValue(omega); // Store the new average omega
        iterator = 0; // Reset iterator after updating omega
    }

    // Calculates the new value for omega considering the current and ideal distances
    private double calculateNewOmega(double obtainedDist) {
        double newOmega = omega + ((omega / obtainedDist * idealDist.idealDist) - omega);
        return Math.min(omegaMax, Math.max(newOmega, omegaMin)); // Ensure omega is within bounds
    }

    public void setDistance(double distLS) {
        iterator++;
        meanLSDist.setValue(distLS); // Update the mean LSDist with the new distance

        // Update omega at specified intervals based on iterations
        if (iterator % numIterUpdate == 0) {
            updateOmega();
        }
    }

    public double getActualOmega() {
        actualOmega = Math.min(omegaMax, Math.max(omega, omegaMin));
        return actualOmega; // Return the current omega ensuring it's within the limits
    }

    @Override
    public String toString() {
        return String.format("o%s: %s meanLSDist%s: %s dMI%s: %s actualOmega: %s obtainedDist: %s averageOmega%s: %s",
            perturbationType.toString().substring(4), decimalFormat.format(omega),
            perturbationType.toString().substring(4), meanLSDist,
            perturbationType.toString().substring(4), decimalFormat.format(idealDist.idealDist),
            decimalFormat.format(getActualOmega()), obtainedDist,
            perturbationType.toString().substring(4), averageOmega);
    }

    public Mean getAverageOmega() {
        return averageOmega; // Accessor for average omega
    }

    public PerturbationType getPerturbationType() {
        return perturbationType; // Accessor for the perturbation type
    }

    public void setActualOmega(double actualOmega) {
        this.actualOmega = actualOmega; // Setter for the actual omega
    }
}