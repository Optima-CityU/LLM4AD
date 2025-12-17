import os

template_program = '''
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
'''

task_description = """
Perform method-level improvements on the Java class `OmegaAdjustment`.
This class acts as the core **Omega Adjustment Mechanism** to adjust the hyperparameter omega_k for perturbtion algorithms within a metaheuristic optimization framework for the CVRP.
Your Ultimate Goal is to refactor this acceptance criterion to enhance the solver's ability to find high-quality solutions in complex CVRP instances.
"""

java_dir = "CVRPLIB_2025"  # 多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
aim_java_relative_path = os.path.join('src', 'AILS-II_origin','src', 'DiversityControl', 'OmegaAdjustment.java')          # 被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"

# java_dir = "CVRPLIB_2025_AILSII"        # 多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
# aim_java_relative_path = os.path.join('Method', 'AILS-II','src', 'DiversityControl', 'DistAdjustment.java')       # 被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"



