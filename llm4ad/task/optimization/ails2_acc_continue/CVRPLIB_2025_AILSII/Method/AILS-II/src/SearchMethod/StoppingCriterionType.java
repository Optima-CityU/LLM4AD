/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package SearchMethod;

public enum StoppingCriterionType 
{
	Time(1),
	Iteration(2);
	
	final int type;
	
	StoppingCriterionType(int type)
	{
		this.type=type;
	}

}
