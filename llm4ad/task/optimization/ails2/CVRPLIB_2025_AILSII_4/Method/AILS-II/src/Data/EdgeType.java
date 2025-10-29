/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Data;

public enum EdgeType 
{
	EUC_2D(0),
	Coord(1),
	EXPLICIT(2);
	
	final int type;
	
	EdgeType(int type)
	{
		this.type=type;
	}

}
