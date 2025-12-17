/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package SearchMethod;

import java.util.Random;

import Data.Instance;
import Solution.Node;
import Solution.Route;
import Solution.Solution;


public class ConstructSolution 
{
	private Route routes[];
	private double f=0;
	private int numRoutes;
	private Node []solution;
	protected Random rand=new Random();
	protected int size;
	Instance instance;
	Node notInserted[];
	int countNotInserted=0;
	
	public ConstructSolution(Instance instance,Config config)
	{
		this.instance=instance;
		this.routes=new Route[instance.getMaxNumberRoutes()];
		this.size=instance.getSize()-1;
		this.notInserted=new Node[size];
	}
	
	private void setSolution(Solution solution) 
	{
		this.numRoutes=solution.numRoutes;
		this.solution=solution.getSolution();
		this.f=solution.f;
		for (int i = 0; i < routes.length; i++) 
			this.routes[i]=solution.routes[i];
	}

	private void assignResult(Solution solution) 
	{
		solution.numRoutes=this.numRoutes;
		solution.f=this.f;
		for (int i = 0; i < routes.length; i++) 
			solution.routes[i]=this.routes[i];
	}

	public void construct(Solution s)
	{
		setSolution(s);
		
		for (int i = 0; i < routes.length; i++)
			routes[i].clean();
		
		int index;
		Node node,bestNode;
		f=0;
		
		for (int i = 0; i < size; i++) 
			notInserted[countNotInserted++]=solution[i];
		
		for (int i = 0; i < numRoutes; i++)
		{
			index=rand.nextInt(countNotInserted);
			f+=routes[i].addNodeEndRoute(notInserted[index]);
			
			node=notInserted[index];
			notInserted[index]=notInserted[countNotInserted-1];
			notInserted[--countNotInserted]=node;
		}
		
		while(countNotInserted>0) 
		{
			index=rand.nextInt(countNotInserted);
			node=notInserted[index];
			bestNode=getBestNoRoutes(node);
			f+=bestNode.route.addAfter(node, bestNode);
			notInserted[index]=notInserted[countNotInserted-1];
			notInserted[--countNotInserted]=node;
		}
		
		assignResult(s);
		s.removeEmptyRoutes();
	}
	
	protected Node getBestKNNNo(Node no)
	{
		double bestCost=Double.MAX_VALUE;
		Node aux,bestNode=null;
		double cost,costPrev;
		
		for (int i = 0; i < solution.length; i++) 
		{
			aux=solution[i];
			if(aux.nodeBelong)
			{
				cost=instance.dist(aux.name,no.name)+instance.dist(no.name,aux.next.name)-instance.dist(aux.name,aux.next.name);
				if(cost<bestCost)
				{
					bestCost=cost;
					bestNode=aux;
				}
			}
		}
		cost=instance.dist(bestNode.name,no.name)+instance.dist(no.name,bestNode.next.name)-instance.dist(bestNode.name,bestNode.next.name);
		costPrev=instance.dist(bestNode.prev.name,no.name)+instance.dist(no.name,bestNode.name)-instance.dist(bestNode.prev.name,bestNode.name);
		if(cost<costPrev)
			return bestNode;
		
		return bestNode.prev;
	}
	
	protected Node getBestNoRoutes(Node no)
	{
		double bestCost=Double.MAX_VALUE;
		Node aux,bestNode=null;
		
		for (int i = 0; i < numRoutes; i++) 
		{
			aux=routes[i].findBestPosition(no);
			if(routes[i].lowestCost<bestCost)
			{
				bestCost=routes[i].lowestCost;
				bestNode=aux;
			}
		}
		
		return bestNode;
	}
	
}
