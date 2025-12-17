/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Perturbation;

public enum PerturbationType 
{
	Sequential(0),
	Concentric(1);
	
	final int type;
	
	PerturbationType(int type)
	{
		this.type=type;
	}

}
