/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Evaluators;

import Solution.Node;

public class FeasibilityEvaluation 
{
	int capViolation=0;
	int capViolationMove=0;
	 
//	-------------------------------SHIFT------------------------------

	public int gainSHIFT(Node a, Node b)
	{
		 capViolation=0;
		 capViolationMove=0;
		 
		 if(a.route.availableCapacity()<0)
			capViolation+=a.route.availableCapacity();
		 
		 if(b.route.availableCapacity()<0)
			capViolation+=b.route.availableCapacity();
		 
		 if((a.route.availableCapacity()+a.demand)<0)
			 capViolationMove+=(a.route.availableCapacity()+a.demand);
			 
		 if((b.route.availableCapacity()-a.demand)<0)
			 capViolationMove+=(b.route.availableCapacity()-a.demand);
		 
		 return -capViolation+capViolationMove;
	}
	
//	-------------------------------SWAP------------------------------
	
	public int gainSWAP(Node a, Node b)
	{
		 capViolation=0;
		 capViolationMove=0;
		 
		 if(a.route.availableCapacity()<0)
			capViolation+=a.route.availableCapacity();
		 
		 if(b.route.availableCapacity()<0)
			capViolation+=b.route.availableCapacity();
		 
		 if((a.route.availableCapacity()+(a.demand-b.demand))<0)
			 capViolationMove+=(a.route.availableCapacity()+(a.demand-b.demand));
			 
		 if((b.route.availableCapacity()-(a.demand-b.demand))<0)
			 capViolationMove+=(b.route.availableCapacity()-(a.demand-b.demand));
		 
		 return -capViolation+capViolationMove;
	}
	
//	-------------------------------CROSS------------------------------

	
	public int gainCross(Node a, Node b)
	{
		 capViolation=0;
		 capViolationMove=0;
		 
		 if(a.route.availableCapacity()<0)
			capViolation+=a.route.availableCapacity();
		 
		 if(b.route.availableCapacity()<0)
			capViolation+=b.route.availableCapacity();
		 
		 if(a.route.availableCapacity()+((a.route.totalDemand-a.accumulatedDemand)-(b.route.totalDemand-b.accumulatedDemand))<0)
			 capViolationMove+=a.route.availableCapacity()+((a.route.totalDemand-a.accumulatedDemand)-(b.route.totalDemand-b.accumulatedDemand));
			 
		 if(b.route.availableCapacity()-((a.route.totalDemand-a.accumulatedDemand)-(b.route.totalDemand-b.accumulatedDemand))<0)
			 capViolationMove+=b.route.availableCapacity()-((a.route.totalDemand-a.accumulatedDemand)-(b.route.totalDemand-b.accumulatedDemand));
		 
		 return -capViolation+capViolationMove;
	}
	
	public int gainCrossInverted(Node a, Node b)
	{
		 capViolation=0;
		 capViolationMove=0;
		 
		 if(a.route.availableCapacity()<0)
			capViolation+=a.route.availableCapacity();
		 
		 if(b.route.availableCapacity()<0)
			capViolation+=b.route.availableCapacity();
		 
		 if(a.route.availableCapacity()+((a.route.totalDemand-a.accumulatedDemand)-b.accumulatedDemand)<0)
			 capViolationMove+=a.route.availableCapacity()+((a.route.totalDemand-a.accumulatedDemand)-b.accumulatedDemand);
			 
		 if(b.route.availableCapacity()-((a.route.totalDemand-a.accumulatedDemand)-b.accumulatedDemand)<0)
			 capViolationMove+=b.route.availableCapacity()-((a.route.totalDemand-a.accumulatedDemand)-b.accumulatedDemand);
		 
		 return -capViolation+capViolationMove;
	}
	 
}
