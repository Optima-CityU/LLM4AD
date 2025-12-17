/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Perturbation;

public enum InsertionHeuristic 
{
	Distance(1),
	Cost(2);
	
	final int heuristic;
	
	InsertionHeuristic(int heuristic)
	{
		this.heuristic=heuristic;
	}
}
