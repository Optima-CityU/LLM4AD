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
        this.omegaMin = 1;
        this.omegaMax = size - 2;
        this.averageOmega = new Mean(numIterUpdate);
        this.meanLSDist = new Mean(numIterUpdate);
        this.idealDist = idealDist;
    }

    // Main method to update omega based on the dynamic average of LSDist and idealDist.
    private void updateOmega() {
        obtainedDist = meanLSDist.getDynamicAverage();
        // Compute the new omega based on obtained distance and ideal distance.
        omega += (omega / obtainedDist * idealDist.idealDist) - omega; 
        omega = clampOmega(omega); // Ensure omega stays within the defined limits.
        averageOmega.setValue(omega);
        iterator = 0; // Reset iterator for the next omega update cycle.
    }
    
    // Clamp omega to stay within the defined minimum and maximum limits.
    private double clampOmega(double value) {
        return Math.min(omegaMax, Math.max(value, omegaMin));
    }

    // Increment the iterator and set the distance for updating the mean LSDist.
    public void setDistance(double distLS) {
        iterator++;
        meanLSDist.setValue(distLS);

        // Update omega at specified intervals.
        if (iterator % numIterUpdate == 0) {
            updateOmega();
        }
    }

    // Get the current valid omega value clamped within the limits.
    public double getActualOmega() {
        actualOmega = clampOmega(omega);
        return actualOmega;
    }

    @Override
    public String toString() {
        return String.format("Perturbation Type: %s, Omega: %s, Mean LSDist: %s, Ideal Distance: %s, " +
                             "Actual Omega: %s, Obtained Distance: %s, Average Omega: %s",
                             perturbationType.toString().substring(4),
                             decimalFormat.format(omega),
                             meanLSDist,
                             decimalFormat.format(idealDist.idealDist),
                             decimalFormat.format(getActualOmega()),
                             obtainedDist,
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
}