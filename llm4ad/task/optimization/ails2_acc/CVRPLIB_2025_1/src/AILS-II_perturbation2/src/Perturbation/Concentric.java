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

//Concentric removal removes all nodes at once and then adds them into the route
public class Concentric extends Perturbation
{
	public Concentric(Instance instance, Config config,
	HashMap<String, OmegaAdjustment> omegaSetup, IntraLocalSearch intraLocalSearch)
	{
		super(instance, config, omegaSetup,intraLocalSearch);
		this.perturbationType=PerturbationType.Concentric;
	}

	public void applyPerturbation(Solution s)
	{
		setSolution(s);
		
//		---------------------------------------------------------------------
		Node reference=solution[rand.nextInt(solution.length)];
		for (int i = 0; i < omega&&i < reference.knn.length&&countCandidates<omega; i++) 
		{
			if(reference.knn[i]!=0)
			{
				node=solution[reference.knn[i]-1];
				candidates[countCandidates]=node;
				countCandidates++;
				node.prevOld=node.prev;
				node.nextOld=node.next;
				f+=node.route.remove(node);
			}
		}
		
		setOrder();
		
		addCandidates();
		
		assignSolution(s);
	}
}
