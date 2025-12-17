/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Improvement;

import java.util.Arrays;

import Data.Instance;
import Evaluators.CostEvaluation;
import Evaluators.FeasibilityEvaluation;
import Evaluators.ExecuteMovement;
import Evaluators.MovementType;
import SearchMethod.Config;
import Solution.Node;
import Solution.Route;
import Solution.Solution;

public class FeasibilityPhase 
{
	private Route routes[];
	private  CandidateNode improvedMoves[];
	private  CandidateNode improvedMatrix[][];

	private CandidateNode improvedNode;

	private int topBest=0;
	private int numRoutes;
	
	Node bestPrevNoRouteI,bestPrevNoRouteJ;

	double f=0;
	
	Node auxOut,auxIn;
	int gain;
	double cost;
	double evaluationCost;
	Route routeA,routeB;
	
	CostEvaluation evaluateCost;
	FeasibilityEvaluation feasibilityEvaluation;
	ExecuteMovement executeMovement;
	Node solution[];
	int limitAdj;
	IntraLocalSearch intraLocalSearch;
	double epsilon;
	int activeNodesCounter;
	
	public FeasibilityPhase(Instance instance,Config config, IntraLocalSearch intraLocalSearch)
	{
		this.evaluateCost=new CostEvaluation(instance);
		this.feasibilityEvaluation=new FeasibilityEvaluation();
		this.executeMovement=new ExecuteMovement(instance);
		this.improvedMoves=new CandidateNode[instance.getMaxNumberRoutes()*(instance.getMaxNumberRoutes()-1)/2];
		this.improvedMatrix=new CandidateNode[instance.getMaxNumberRoutes()][instance.getMaxNumberRoutes()];
		
		for (int i = 0; i < improvedMatrix.length; i++)
		{
			for (int j = 0; j < improvedMatrix.length; j++) 
			{
				improvedMatrix[i][j]=new CandidateNode(evaluateCost);
				improvedMatrix[j][i]=improvedMatrix[i][j];
			}
		}
		
		this.limitAdj=Math.min(config.getVarphi(), instance.getSize()-1);
		
		this.intraLocalSearch=intraLocalSearch;
		this.epsilon=config.getEpsilon();
	}
	
	private boolean feasible() 
	{
		for (int i = 0; i < numRoutes; i++)
		{
			if(routes[i].availableCapacity()<0)
				return false;
		}
		return true;
	}
	
	private void setSolution(Solution solution) 
	{
		this.numRoutes=solution.numRoutes;
		this.solution=solution.getSolution();
		this.f=solution.f;
		this.routes=solution.routes;
	}

	private void assignResult(Solution solution) 
	{
		solution.numRoutes=this.numRoutes;
		solution.f=this.f;
	}

	public void makeFeasible(Solution solution)
	{
		setSolution(solution);
		boolean feasible=false;
		
		do
		{
			topBest=0;
			for (int i = 0; i < numRoutes; i++) 
				routes[i].setAccumulatedDemand();
			
			for (int i = 0; i < numRoutes; i++) 
			{
				if(routes[i].modified)
					browseRoutes(routes[i]); 
			}
			
			if(topBest>0)
				execute();

			if(feasible())
			{
				intraLocalSearch();
				
				assignResult(solution);
				solution.removeEmptyRoutes();
				feasible=true;
			}
			else
			{
				numRoutes++;
				routes[numRoutes-1].clean();
			}
		}
		while(!feasible);
		
	}
	
	//-----------------------------------Factibilizando com o Best Improviment-------------------------------------------
	
	private void execute() 
	{
		while(topBest>0)
		{
			Arrays.sort(improvedMoves,0,topBest);
			
			routeA=improvedMoves[0].routeA;
			routeB=improvedMoves[0].routeB;
			f+=executeMovement.apply(improvedMoves[0]);
			
			intraLocalSearch(routeA);
			intraLocalSearch(routeB);
			
			routeA.setAccumulatedDemand();
			routeB.setAccumulatedDemand();
			
			activeNodesCounter=0;
			for (int k = 0; k < topBest; k++) 
			{
				if(improvedMoves[k].routeA==routeA||improvedMoves[k].routeA==routeB||improvedMoves[k].routeB==routeA||improvedMoves[k].routeB==routeB)
				{
					improvedMoves[k].clean();
					activeNodesCounter++;
				}
			}
	
			Arrays.sort(improvedMoves,0,topBest);
			topBest-=activeNodesCounter;
			
			browseRoutes(routeA);
			browseRoutes(routeB);
		}
	}
	
	public void browseRoutes(Route route)
	{
		if(route.getNumElements()>1)
		{
			searchBestSHIFT(route);
			searchBestSwapStarKnn(route);
			searchBestCross(route);
		}
	}
	
	public void calculateCost()
	{
		if(cost>=0)
			evaluationCost=((double)cost+1)/gain;
		else
			evaluationCost=(double)cost/gain;
	}
	
	private void searchBestSwapStarKnn(Route route)
	{
		auxOut=route.first.next;
		do
		{
			if(auxOut.modified)
			{
				for (int j = 0; j < limitAdj; j++) 
				{
					if(auxOut.getKnn()[j]!=0&&auxOut.route.isFeasible()^solution[auxOut.getKnn()[j]-1].route.isFeasible())
					{
						auxIn=solution[auxOut.getKnn()[j]-1];
						gain=feasibilityEvaluation.gainSWAP(auxOut, auxIn);
						if(gain>0)
						{
							
							bestPrevNoRouteI=auxIn.route.findBestPositionExceptAfterNodeKNN(auxOut,auxIn,solution);
							bestPrevNoRouteJ=auxOut.route.findBestPositionExceptAfterNodeKNN(auxIn,auxOut,solution);
							
							cost=evaluateCost.costSwapStar(auxOut,auxIn,bestPrevNoRouteI,bestPrevNoRouteJ);
							calculateCost();
							
							improvedNode=improvedMatrix[auxOut.route.nameRoute][auxIn.route.nameRoute];
							
							if(evaluationCost<improvedNode.evaluationCost)
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.SWAPStar, auxOut, auxIn,bestPrevNoRouteI, bestPrevNoRouteJ, evaluationCost, gain);
							}
						}
					}
				}
			}
			auxOut=auxOut.next;
		}
		while(auxOut!=route.first);
	}
	
	private void searchBestSHIFT(Route route)
	{
		auxOut=route.first.next;
		do
		{
			if(auxOut.modified)
			{
				for (int i = 0; i < numRoutes; i++) 
				{
					if(!auxOut.route.isFeasible()&&routes[i].isFeasible())
					{
						auxIn=routes[i].first;
						gain=feasibilityEvaluation.gainSHIFT(auxOut, auxIn);
						if(gain>0)
						{
							cost=evaluateCost.costSHIFT(auxOut, auxIn);
							calculateCost();
							improvedNode=improvedMatrix[auxOut.route.nameRoute][auxIn.route.nameRoute];
							
							if(evaluationCost<improvedNode.evaluationCost)
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.SHIFT, auxOut, auxIn,evaluationCost,gain);
							}
						}
					}
				}
				
				for (int j = 0; j < limitAdj; j++) 
				{
					if(auxOut.getKnn()[j]!=0&&!auxOut.route.isFeasible()&&solution[auxOut.getKnn()[j]-1].route.isFeasible())
					{
						auxIn=solution[auxOut.getKnn()[j]-1];
						gain=feasibilityEvaluation.gainSHIFT(auxOut, auxIn);
						if(gain>0)
						{
							cost=evaluateCost.costSHIFT(auxOut, auxIn);
							calculateCost();
							improvedNode=improvedMatrix[auxOut.route.nameRoute][auxIn.route.nameRoute];
							
							if(evaluationCost<improvedNode.evaluationCost)
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.SHIFT, auxOut, auxIn,evaluationCost,gain);
							}
						}
					}
					
					if(auxOut.getKnn()[j]!=0&&auxOut.route.isFeasible()&&!solution[auxOut.getKnn()[j]-1].route.isFeasible())
					{
						auxIn=solution[auxOut.getKnn()[j]-1];
						gain=feasibilityEvaluation.gainSHIFT(auxIn, auxOut);
						if(gain>0)
						{
							cost=evaluateCost.costSHIFT(auxIn, auxOut);
							calculateCost();
							
							improvedNode=improvedMatrix[auxOut.route.nameRoute][auxIn.route.nameRoute];
							
							if(evaluationCost<improvedNode.evaluationCost)
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.SHIFT, auxIn ,auxOut ,evaluationCost,gain);
							}
						}
					}
				}
			}
			auxOut=auxOut.next;
		}
		while(auxOut!=route.first);
	}
	
	
	private void searchBestCross(Route route)
	{
		auxOut=route.first;
		do
		{
			if(auxOut.modified)
			{
				for (int i = 0; i < numRoutes; i++) 
				{
					if(auxOut.route.isFeasible()^routes[i].isFeasible())
					{
						auxIn=routes[i].first;
						gain=feasibilityEvaluation.gainCross(auxOut, auxIn);
						if(gain>0)
						{
							cost=evaluateCost.costCross(auxOut, auxIn);
							calculateCost();
							
							improvedNode=improvedMatrix[auxOut.route.nameRoute][auxIn.route.nameRoute];
							
							if(evaluationCost<improvedNode.evaluationCost)
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.Cross, auxOut, auxIn,evaluationCost,gain);
							}
							
						}
						gain=feasibilityEvaluation.gainCrossInverted(auxOut, auxIn);
						if(gain>0)
						{
							cost=evaluateCost.inversedCostCross(auxOut, auxIn);
							calculateCost();
							
							improvedNode=improvedMatrix[auxOut.route.nameRoute][auxIn.route.nameRoute];
							
							if(evaluationCost<improvedNode.evaluationCost)
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.CrossInverted, auxOut, auxIn,evaluationCost,gain);
							}
						}
					}
				}
				
				for (int j = 0; j < limitAdj; j++) 
				{
					if(auxOut.getKnn()[j]!=0&&auxOut.route.isFeasible()^solution[auxOut.getKnn()[j]-1].route.isFeasible())
					{
						auxIn=solution[auxOut.getKnn()[j]-1];
						gain=feasibilityEvaluation.gainCross(auxOut, auxIn);
						if(gain>0)
						{
							cost=evaluateCost.costCross(auxOut, auxIn);
							calculateCost();
							improvedNode=improvedMatrix[auxOut.route.nameRoute][auxIn.route.nameRoute];
							
							if(evaluationCost<improvedNode.evaluationCost)
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.Cross, auxOut, auxIn,evaluationCost,gain);
							}
						}
						gain=feasibilityEvaluation.gainCrossInverted(auxOut, auxIn);
						if(gain>0)
						{
							cost=evaluateCost.inversedCostCross(auxOut, auxIn);
							calculateCost();
							improvedNode=improvedMatrix[auxOut.route.nameRoute][auxIn.route.nameRoute];
							
							if(evaluationCost<improvedNode.evaluationCost)
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.CrossInverted, auxOut, auxIn,evaluationCost,gain);
							}
						}
					}
				}
			}
			auxOut=auxOut.next;
		}
		while(auxOut!=route.first);
	}
	
	private void intraLocalSearch(Route route)
	{
		f+=intraLocalSearch.intraLocalSearch(route, solution);
	}
	
	private void intraLocalSearch()
	{
		for (int i = 0; i < numRoutes; i++)
		{
			if(routes[i].modified)
				f+=intraLocalSearch.intraLocalSearch(routes[i], solution);
		}
	}
	
}
