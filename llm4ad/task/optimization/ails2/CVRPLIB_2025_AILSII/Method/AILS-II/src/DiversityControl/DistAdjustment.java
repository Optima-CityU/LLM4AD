/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
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
