/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Improvement;

import Data.Instance;
import Evaluators.CostEvaluation;
import Evaluators.ExecuteMovement;
import Evaluators.MovementType;
import SearchMethod.Config;
import Solution.Node;
import Solution.Route;

public class IntraLocalSearch 
{
	private  CandidateNode improve;
	Node first;
	int numElements=0;
	MovementType moveType;
	int iterator;
	double lowestCost;
	double prevF;
	Node auxOut,auxIn;
	double cost;
	boolean changeFlag=false;
	CostEvaluation evaluateCost;
	ExecuteMovement executeMovement;
	int limitAdj;
	
	public IntraLocalSearch(Instance instance,Config config)
	{
		this.evaluateCost=new CostEvaluation(instance);
		this.executeMovement=new ExecuteMovement(instance);
		this.improve=new CandidateNode(evaluateCost);
		this.limitAdj=config.getVarphi();
	}
	
	private void setRoute(Route route,Node solution[]) 
	{
		this.prevF=route.fRoute;
		this.first=route.first;
		this.numElements=route.numElements;
	}

	public double intraLocalSearch(Route route,Node solution[])
	{
		setRoute(route,solution);
		
		iterator=0;
		changeFlag=true;
		while(changeFlag)
		{
			iterator++;
			changeFlag=false;
			lowestCost=0;
			auxOut=first;
			do
			{
				if(auxOut.modified)
				{
					for (int j = 0; j < limitAdj; j++) 
					{
						if(auxOut.getKnn()[j]==0)
							auxIn=first;
						else
							auxIn=solution[auxOut.getKnn()[j]-1];
						
						if(auxOut.route.nameRoute==auxIn.route.nameRoute)
						{
							//2Opt
							if(auxOut!=auxIn&&auxIn!=auxOut.next)
							{
								cost=evaluateCost.cost2Opt(auxOut,auxIn);
								if(lowestCost>cost)
								{
									lowestCost=cost;
									moveType=MovementType.TwoOpt;
									improve.setImprovedNode(lowestCost, moveType, auxOut, auxIn,0,0,lowestCost);
									changeFlag=true;
								}
							}
							
							//SHIFT
							if(numElements>2&&auxOut!=auxIn&&auxOut!=auxIn.next)
							{
								cost=evaluateCost.costSHIFT(auxOut,auxIn);
								if(lowestCost>cost)
								{
									lowestCost=cost;
									moveType=MovementType.SHIFT;
									improve.setImprovedNode(lowestCost, moveType, auxOut, auxIn,0,0,lowestCost);
									changeFlag=true;
								}
								
								if(auxIn!=auxOut.next)
								{
									cost=evaluateCost.costSHIFT(auxIn,auxOut);
									if(lowestCost>cost)
									{
										lowestCost=cost;
										moveType=MovementType.SHIFT;
										improve.setImprovedNode(lowestCost, moveType, auxIn, auxOut,0,0,lowestCost);
										changeFlag=true;
									}
								}
							}
							
							//SWAP
							if(numElements>2&&auxIn!=auxOut&&auxIn.next!=auxOut)
							{
								cost=evaluateCost.costSWAP(auxOut,auxIn);
								if(lowestCost>cost)
								{
									lowestCost=cost;
									moveType=MovementType.SWAP;
									improve.setImprovedNode(lowestCost, moveType, auxOut, auxIn,0,0,lowestCost);
									changeFlag=true;
								}
							}
						}
					}
				}
				
				auxOut=auxOut.next;
			}
			while(auxOut!=first);
			
			if(changeFlag)
				executeMovement.apply(improve);
		}
		
		return route.fRoute-prevF;
	}
	
}
