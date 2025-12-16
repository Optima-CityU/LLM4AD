/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Evaluators;

import Data.Instance;
import Solution.Node;

public class CostEvaluation 
{
	Instance instance;
	double cost;
	public CostEvaluation(Instance instance)
	{
		this.instance=instance;
	}
	
//	-------------------------------SHIFT------------------------------
	
	public double costSHIFT(Node a, Node b)
	{
		 return instance.dist(a.prev.name,a.next.name)-instance.dist(a.name,a.prev.name)-instance.dist(a.name,a.next.name)+
				instance.dist(a.name,b.name)+instance.dist(a.name,b.next.name)-instance.dist(b.name,b.next.name);
	}
		
//	-------------------------------SWAP------------------------------
	
	public double costSWAP(Node a, Node b)
	{
		if(a.next!=b&&a.prev!=b)
		{
			return 	-(instance.dist(a.name,a.prev.name)+instance.dist(a.name,a.next.name)+
					instance.dist(b.name,b.prev.name)+instance.dist(b.name,b.next.name))+				
					(instance.dist(a.name,b.prev.name)+instance.dist(a.name,b.next.name)+
					instance.dist(b.name,a.prev.name)+instance.dist(b.name,a.next.name));
		}
		else
		{
			if(a.next==b)
				return 	-(instance.dist(a.name,a.prev.name)+instance.dist(b.name,b.next.name))+
					(instance.dist(a.name,b.next.name)+instance.dist(b.name,a.prev.name));
			else
				return 	-(instance.dist(b.name,b.prev.name)+instance.dist(a.name,a.next.name))+
						(instance.dist(b.name,a.next.name)+instance.dist(a.name,b.prev.name));
		}
	}
	
	public double costSwapStar(Node a, Node b,Node prevA, Node prevB)
	{
		if(prevA.next.name!=b.name&&prevB.next.name!=a.name)
		{
			return costSHIFT(a,prevA)+costSHIFT(b,prevB);
		}
		else 
		{
			if(prevA.next.name==b.name&&prevB.next.name!=a.name)
			{
				return	-instance.dist(prevA.name,b.name)
						-instance.dist(b.name,b.next.name)
						+instance.dist(prevA.name,a.name)
						+instance.dist(b.next.name,a.name)
						
						-instance.dist(prevB.name,prevB.next.name)
						+instance.dist(prevB.name,b.name)
						+instance.dist(b.name,prevB.next.name)
						
						-instance.dist(a.prev.name,a.name)
						-instance.dist(a.name,a.next.name)
						+instance.dist(a.prev.name,a.next.name);
						
			}
			
			if(prevA.next.name!=b.name&&prevB.next.name==a.name)
			{
				return	-instance.dist(prevB.name,a.name)
						-instance.dist(a.name,a.next.name)
						+instance.dist(prevB.name,b.name)
						+instance.dist(a.next.name,b.name)
						
						-instance.dist(prevA.name,prevA.next.name)
						+instance.dist(prevA.name,a.name)
						+instance.dist(a.name,prevA.next.name)
						
						-instance.dist(b.prev.name,b.name)
						-instance.dist(b.name,b.next.name)
						+instance.dist(b.prev.name,b.next.name);
			}
			
			return costSWAP(a, b);
		}
	}
	
//		-------------------------------CROSS------------------------------

	public double costCross(Node a, Node b)
	{
		 return -(instance.dist(a.name,a.next.name)+instance.dist(b.name,b.next.name))
				 +(instance.dist(a.name,b.next.name)+instance.dist(b.name,a.next.name));
	}
	 
	public double inversedCostCross(Node a, Node b)
	{
		 return -(instance.dist(a.name,a.next.name)+instance.dist(b.name,b.next.name))
				 +(instance.dist(a.name,b.name)+instance.dist(b.next.name,a.next.name));
	}
	 
	public double cost2Opt(Node a, Node b)
	{
		return 	-(instance.dist(a.name,a.next.name)+instance.dist(b.name,b.next.name))+				
				(instance.dist(a.name,b.name)+instance.dist(a.next.name,b.next.name));
	}
	 
}
