/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package DiversityControl;

import java.text.DecimalFormat;
import java.util.Random;

import Auxiliary.Mean;
import Perturbation.PerturbationType;
import SearchMethod.Config;

public class OmegaAdjustment 
{
	double omega,omegaMin,omegaMax;
	Mean meanLSDist;
	int iterator=0;
	DecimalFormat deci2=new DecimalFormat("0.00");
	double obtainedDist;
	
	Mean averageOmega;
	
	double actualOmega;
	Random rand=new Random();
	PerturbationType perturbationType;
	int numIterUpdate;
	IdealDist idealDist;
	
	public OmegaAdjustment(PerturbationType perturbationType, Config config, Integer size,IdealDist idealDist) 
	{
		this.perturbationType = perturbationType;
		this.omega = idealDist.idealDist;
		this.numIterUpdate = config.getGamma();
		this.omegaMin=1;
		this.omegaMax=size-2;
		this.averageOmega=new Mean(config.getGamma());
		this.meanLSDist=new Mean(config.getGamma());

		this.idealDist=idealDist;
	}
	
	public void setupOmega()
	{
		obtainedDist=meanLSDist.getDynamicAverage();

		omega+=((omega/obtainedDist*idealDist.idealDist)-omega);

		omega=Math.min(omegaMax, Math.max(omega, omegaMin));
		
		averageOmega.setValue(omega);
		
		iterator=0;
	}
	
	public void setDistance(double distLS)
	{
		iterator++;
		
		meanLSDist.setValue(distLS);

		if(iterator%numIterUpdate==0)
			setupOmega();
	}
	
	public double getActualOmega() 
	{
		actualOmega=omega;
		actualOmega=Math.min(omegaMax, Math.max(actualOmega, omegaMin));
		return actualOmega;
	}

	@Override
	public String toString() {
		return 
		"o"+String.valueOf(perturbationType).substring(4)+": " + deci2.format(omega) 
		+ " meanLSDist"+String.valueOf(perturbationType).substring(4)+": " + meanLSDist
		+ " dMI"+String.valueOf(perturbationType).substring(4)+": " + deci2.format(idealDist.idealDist)
		+ " actualOmega: "+deci2.format(actualOmega)
		+ " obtainedDist: "+obtainedDist
		+ " averageOmega"+String.valueOf(perturbationType).substring(4)+": " + averageOmega;
	}
	
	public Mean getAverageOmega() 
	{
		return averageOmega;
	}

	public PerturbationType getPerturbationType() {
		return perturbationType;
	}

	public void setActualOmega(double actualOmega) {
		this.actualOmega = actualOmega;
	}

}
