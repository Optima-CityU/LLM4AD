package DiversityControl;

import java.text.DecimalFormat;
import java.util.Random;

import Auxiliary.Mean;
import Perturbation.PerturbationType;
import SearchMethod.Config;

public class OmegaAdjustment {
    private double omega, omegaMin, omegaMax;
    private final Mean meanLSDist;
    private int iterator = 0;
    private final DecimalFormat decimalFormat = new DecimalFormat("0.00");
    private double obtainedDist;
    
    private final Mean averageOmega;
    private final Random rand = new Random();
    private final PerturbationType perturbationType;
    private final int numIterUpdate;
    private final IdealDist idealDist;

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
    
    // Updates omega based on the adjusted mean LSDist relative to idealDist
    private void updateOmega() {
        obtainedDist = meanLSDist.getDynamicAverage();
        double newOmega = omega * (idealDist.idealDist / obtainedDist);
        omega = clamp(newOmega, omegaMin, omegaMax);
        averageOmega.setValue(omega);
        resetIterator();
    }
    
    // Sets the distance and manages the omega update cycle
    public void setDistance(double distLS) {
        meanLSDist.setValue(distLS);
        iterator++;
        if (shouldUpdateOmega()) {
            updateOmega();
        }
    }

    // Determines if it's time to update omega
    private boolean shouldUpdateOmega() {
        return iterator % numIterUpdate == 0;
    }

    // Clamps the value between a minimum and maximum
    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(value, max));
    }

    // Resets the iterator
    private void resetIterator() {
        iterator = 0;
    }

    // Returns the current valid omega
    public double getActualOmega() {
        return clamp(omega, omegaMin, omegaMax);
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
        return averageOmega;
    }

    public PerturbationType getPerturbationType() {
        return perturbationType;
    }
    
    public void setActualOmega(double actualOmega) {
        this.omega = clamp(actualOmega, omegaMin, omegaMax);
    }
}