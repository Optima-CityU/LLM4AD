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


public class LocalSearch 
{
	private Route routes[];
	private  CandidateNode improvedMoves[];
	private  CandidateNode improvedMatrix[][];
	private CandidateNode improvedNode;
	private int topBest=0;
	private int numRoutes;
	
	Node bestPrevNoRouteI,bestPrevNoRouteJ;

	double f=0;
	
	double cost;
	Node auxRouteI,auxRouteJ;
	Route routeA,routeB;
	CostEvaluation evaluateCost;
	FeasibilityEvaluation feasibilityEvaluation;
	ExecuteMovement executeMovement;
	Node solution[];
	int limitAdj;
	IntraLocalSearch intraLocalSearch;
	double epsilon;
	int activeNodesCounter;
	
	public LocalSearch(Instance instance,Config config, IntraLocalSearch intraLocalSearch)
	{
		this.evaluateCost=new CostEvaluation(instance);
		this.feasibilityEvaluation=new FeasibilityEvaluation();
		this.executeMovement=new ExecuteMovement(instance);
		this.improvedMoves=new CandidateNode[instance.getMaxNumberRoutes()*(instance.getMaxNumberRoutes()-1)/2];
		this.improvedMatrix=new CandidateNode[instance.getMaxNumberRoutes()][instance.getMaxNumberRoutes()];
		
		for (int i = 0; i < improvedMatrix.length; i++)
		{
			for (int j = i+1; j < improvedMatrix.length; j++) 
			{
				improvedMatrix[i][j]=new CandidateNode(evaluateCost);
				improvedMatrix[j][i]=improvedMatrix[i][j];
			}
		}
		
		this.limitAdj=Math.min(config.getVarphi(), instance.getSize()-1);
		this.intraLocalSearch=intraLocalSearch;
		this.epsilon=config.getEpsilon();
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

	public void localSearch(Solution solution,boolean isRouteEmptyflag)
	{
		setSolution(solution);
		topBest=0;
			
		for (int i = 0; i < numRoutes; i++) 
			routes[i].setAccumulatedDemand();
		
		for (int i = 0; i < numRoutes; i++) 
		{
			if(routes[i].numElements>1&&routes[i].modified)
				browseRoutes(routes[i]);
		}
		
		if(topBest>0)
			execute();
		
		assignResult(solution);
		if(isRouteEmptyflag)
			solution.removeEmptyRoutes();
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
				if(improvedMoves[k].routeA==routeA||improvedMoves[k].routeB==routeA||improvedMoves[k].routeA==routeB||improvedMoves[k].routeB==routeB)
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
		if(route.numElements>1)
		{
			searchBestSHIFT(route);
			searchBestSwapStarKnn(route);
			searchBestCross(route);
		}
	}
	
	private void searchBestSwapStarKnn(Route route)
	{
		auxRouteI=route.first.next;
		do
		{
			if(auxRouteI.modified)
			{
				for (int j = 0; j < limitAdj; j++) 
				{
					if(auxRouteI.getKnn()[j]!=0&&solution[auxRouteI.getKnn()[j]-1].route!=route)
					{
						auxRouteJ=solution[auxRouteI.getKnn()[j]-1];
						if(feasibilityEvaluation.gainSWAP(auxRouteI, auxRouteJ)>=0)
						{
							bestPrevNoRouteI=auxRouteJ.route.findBestPositionExceptAfterNodeKNN(auxRouteI,auxRouteJ,solution);
							bestPrevNoRouteJ=auxRouteI.route.findBestPositionExceptAfterNodeKNN(auxRouteJ,auxRouteI,solution);
							
							cost=evaluateCost.costSwapStar(auxRouteI,auxRouteJ,bestPrevNoRouteI,bestPrevNoRouteJ);
							improvedNode=improvedMatrix[auxRouteI.route.nameRoute][auxRouteJ.route.nameRoute];
							
							if(((cost-improvedNode.cost)<-epsilon)&&((cost-0)<-epsilon))
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.SWAPStar, auxRouteI, auxRouteJ,bestPrevNoRouteI, bestPrevNoRouteJ, cost);
							}
						}
					}
				}
			}
			
			auxRouteI=auxRouteI.next;
		}
		while(auxRouteI!=route.first);
	}
	
	private void searchBestSHIFT(Route route)
	{
		auxRouteI=route.first.next;
		do
		{
			if(auxRouteI.modified)
			{
				for (int j = 0; j < limitAdj; j++) 
				{
					if(auxRouteI.getKnn()[j]==0)
					{
						for (int i = 0; i < numRoutes; i++) 
						{
							if(routes[i]!=route)
							{
								auxRouteJ=routes[i].first;
								if(feasibilityEvaluation.gainSHIFT(auxRouteI, auxRouteJ)>=0)
								{
									cost=evaluateCost.costSHIFT(auxRouteI,auxRouteJ);
									improvedNode=improvedMatrix[auxRouteI.route.nameRoute][auxRouteJ.route.nameRoute];
									
									if(((cost-improvedNode.cost)<-epsilon)&&((cost-0)<-epsilon))
									{
										if(!improvedNode.active)
											improvedMoves[topBest++]=improvedNode;
										
										improvedNode.setImprovedNode(cost, MovementType.SHIFT, auxRouteI, auxRouteJ,cost);
									}
								}
							}
						}
					}
					else if(solution[auxRouteI.getKnn()[j]-1].route!=route)
					{
						auxRouteJ=solution[auxRouteI.getKnn()[j]-1];
						if(feasibilityEvaluation.gainSHIFT(auxRouteI, auxRouteJ)>=0)
						{
							cost=evaluateCost.costSHIFT(auxRouteI,auxRouteJ);
							improvedNode=improvedMatrix[auxRouteI.route.nameRoute][auxRouteJ.route.nameRoute];
							
							if(((cost-improvedNode.cost)<-epsilon)&&((cost-0)<-epsilon))
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.SHIFT, auxRouteI, auxRouteJ,cost);
							}
						}
						
						if(feasibilityEvaluation.gainSHIFT(auxRouteJ, auxRouteI)>=0)
						{
							cost=evaluateCost.costSHIFT(auxRouteJ, auxRouteI);
							improvedNode=improvedMatrix[auxRouteI.route.nameRoute][auxRouteJ.route.nameRoute];
							
							if(((cost-improvedNode.cost)<-epsilon)&&((cost-0)<-epsilon))
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.SHIFT, auxRouteJ,auxRouteI, cost);
							}
						}
					}
				}
			}
			
			auxRouteI=auxRouteI.next;
		}
		while(auxRouteI!=route.first);
	}
	
	private void searchBestCross(Route route)
	{
		auxRouteI=route.first;
		do
		{
			if(auxRouteI.modified)
			{
				for (int j = 0; j < limitAdj; j++) 
				{
					if(auxRouteI.getKnn()[j]==0)
					{
						for (int i = 0; i < numRoutes; i++) 
						{
							if(routes[i]!=route)
							{
								auxRouteJ=routes[i].first;
								if(feasibilityEvaluation.gainCross(auxRouteI, auxRouteJ)>=0)
								{
									cost=evaluateCost.costCross(auxRouteI,auxRouteJ);
									improvedNode=improvedMatrix[auxRouteI.route.nameRoute][auxRouteJ.route.nameRoute];
									
									if(((cost-improvedNode.cost)<-epsilon)&&((cost-0)<-epsilon))
									{
										if(!improvedNode.active)
											improvedMoves[topBest++]=improvedNode;
										
										improvedNode.setImprovedNode(cost, MovementType.Cross, auxRouteI, auxRouteJ,cost);
									}
								}
								
								if(feasibilityEvaluation.gainCrossInverted(auxRouteI, auxRouteJ)>=0)
								{
									cost=evaluateCost.inversedCostCross(auxRouteI,auxRouteJ);
									improvedNode=improvedMatrix[auxRouteI.route.nameRoute][auxRouteJ.route.nameRoute];
									
									if(((cost-improvedNode.cost)<-epsilon)&&((cost-0)<-epsilon))
									{
										if(!improvedNode.active)
											improvedMoves[topBest++]=improvedNode;
										
										improvedNode.setImprovedNode(cost, MovementType.CrossInverted, auxRouteI, auxRouteJ,cost);
									}
								}
							}
						}
					}
					else if(solution[auxRouteI.getKnn()[j]-1].route!=route)
					{
						auxRouteJ=solution[auxRouteI.getKnn()[j]-1];
						if(feasibilityEvaluation.gainCross(auxRouteI, auxRouteJ)>=0)
						{
							cost=evaluateCost.costCross(auxRouteI,auxRouteJ);
							improvedNode=improvedMatrix[auxRouteI.route.nameRoute][auxRouteJ.route.nameRoute];
							
							if(((cost-improvedNode.cost)<-epsilon)&&((cost-0)<-epsilon))
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.Cross, auxRouteI, auxRouteJ,cost);
							}
						}
						if(feasibilityEvaluation.gainCrossInverted(auxRouteI, auxRouteJ)>=0)
						{
							cost=evaluateCost.inversedCostCross(auxRouteI,auxRouteJ);
							improvedNode=improvedMatrix[auxRouteI.route.nameRoute][auxRouteJ.route.nameRoute];
							
							if(((cost-improvedNode.cost)<-epsilon)&&((cost-0)<-epsilon))
							{
								if(!improvedNode.active)
									improvedMoves[topBest++]=improvedNode;
								
								improvedNode.setImprovedNode(cost, MovementType.CrossInverted, auxRouteI, auxRouteJ,cost);
							}
						}
					}
				}
			}
			
			auxRouteI=auxRouteI.next;
		}
		while(auxRouteI!=route.first);
	}
	
	private void intraLocalSearch(Route route)
	{
		f+=intraLocalSearch.intraLocalSearch(route, solution);
	}
	
}
