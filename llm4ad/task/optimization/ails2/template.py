import os

template_program = '''
package DiversityControl;

import SearchMethod.Config;
import SearchMethod.StoppingCriterionType;

public class DistAdjustment 
{
	int distMMin;
	int distMMax;
	int iterator;
	long ini;
	double executionMaximumLimit;
	double alpha=1;
	StoppingCriterionType stoppingCriterionType;
	IdealDist idealDist;

	public DistAdjustment(IdealDist idealDist,Config config,double executionMaximumLimit) 
	{
		this.idealDist=idealDist;
		this.executionMaximumLimit=executionMaximumLimit;
		this.distMMin=config.getDMin();
		this.distMMax=config.getDMax();
		this.idealDist.idealDist=distMMax;
		this.stoppingCriterionType=config.getStoppingCriterionType();
	}

	public void distAdjustment()
	{
		if(iterator==0)
			ini=System.currentTimeMillis();
		
		iterator++;
		
		switch(stoppingCriterionType)
		{
			case Iteration: 	iterationAdjustment(); break;
			case Time: timeAdjustment(); break;
			default:
				break;
								
		}
		
		idealDist.idealDist*=alpha;
		idealDist.idealDist= Math.min(distMMax, Math.max(idealDist.idealDist, distMMin));
		
	}
	
	private void iterationAdjustment()
	{
		alpha=Math.pow((double)distMMin/(double)distMMax, (double) 1/executionMaximumLimit);
	}
	
	private void timeAdjustment()
	{
		double current=(double)(System.currentTimeMillis()-ini)/1000;
		double timePercentage=current/executionMaximumLimit;
		double total=(double)iterator/timePercentage;
		alpha=Math.pow((double)distMMin/(double)distMMax, (double) 1/total);
	}
}
'''

task_description = """
Improve a Java class named `DistAdjustment`.
This class is a core component in an optimization or evolutionary search process (e.g., AILS2 for CVRP) 
responsible for **Diversity Control**. Its primary function is to dynamically adjust the **"ideal diversity distance" 
(`idealDist`)** from an initial maximum value (`distMMax`) toward a minimum value (`distMMin`) over the course of 
the search. This adjustment balances **exploration** (high distance) and **exploitation** (low distance) to enhance overall algorithmic efficiency and convergence. 

Please analyze the current design and propose method-level improvements along with the corresponding Java implementation. 
Potential improvement directions include, but are not limited to:
1. Improving the numerical stability and smoothness of the distance decay process;
2. Adding flexibility to support adaptive or nonlinear decay schedules 
   (e.g., exponential, cosine, or piecewise functions);
3. Refactor the decay logic to support a more flexible and robust scheduling mechanism.

## Ultimate Goal
Enhance this diversity-control component to make it more computationally efficient, numerically stable, and readily scalable to large-scale CVRP instances, thereby improving the overall robustness and research extensibility of the optimization framework.
"""

java_dir = "CVRPLIB_2025"        # 多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
aim_java_relative_path = os.path.join('src', 'AILS-II_origin','src', 'DiversityControl', 'DistAdjustment.java')       # 被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"

# java_dir = "CVRPLIB_2025_AILSII"        # 多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
# aim_java_relative_path = os.path.join('Method', 'AILS-II','src', 'DiversityControl', 'DistAdjustment.java')       # 被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"



