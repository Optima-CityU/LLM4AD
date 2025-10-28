/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Data;

public class Best 
{
	private double optimal;

	public Best(double optimal)
	{
		this.optimal = optimal;
	}

	public double getOptimal() {
		return optimal;
	}

	public synchronized void setOptimal(double optimal) {
		this.optimal = optimal;
	}

	
}
