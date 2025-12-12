/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Perturbation;

import java.util.HashMap;

import Data.Instance;
import DiversityControl.OmegaAdjustment;
import Improvement.IntraLocalSearch;
import SearchMethod.Config;
import Solution.Node;
import Solution.Solution;

//Sequential removal of all at once to add the nodes
public class Sequential extends Perturbation
{
	
	public Sequential(Instance instance, Config config, 
	HashMap<String, OmegaAdjustment> omegaSetup, IntraLocalSearch intraLocalSearch)
	{
		super(instance, config,omegaSetup,intraLocalSearch);
		this.perturbationType=PerturbationType.Sequential;
	}

	public void applyPerturbation(Solution s)
	{
		setSolution(s);
		
//		---------------------------------------------------------------------
		int contSizeString;
		double sizeString;
		Node initialNode;
		
		while(countCandidates<(int)omega)
		{
			sizeString=Math.min(Math.max(1, size),(int)omega-countCandidates);
			
			node=solution[rand.nextInt(size)];
			while(!node.nodeBelong)
				node=solution[rand.nextInt(size)];
			
			initialNode=node;
			
			contSizeString=0;
			do
			{
				contSizeString++;
				node=initialNode.next;
				if(node.name==0)
					node=node.next;
				
				candidates[countCandidates++]=node;
				
				node.prevOld=node.prev;
				node.nextOld=node.next;
				
				f+=node.route.remove(node);
			}
			while(initialNode.name!=node.name&&contSizeString<sizeString);
		}
		
		setOrder();
		
		addCandidates();
		
		assignSolution(s);
	}

}
