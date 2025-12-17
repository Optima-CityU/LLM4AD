/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Auxiliary;

import java.text.DecimalFormat;

public class Mean 
{
	int numIterUpdate;
	double dynamicAverage=0,globalAverage;
	int iterations=0;
	DecimalFormat deci2=new DecimalFormat("0.00");

	public Mean(int numIterUpdate)
	{
		this.numIterUpdate=numIterUpdate;
	}
	
	public void setValue(double value)
	{
		iterations++;
		if(iterations<numIterUpdate)
			dynamicAverage=(dynamicAverage*(iterations-1)+value)/iterations;
		else
			dynamicAverage=((dynamicAverage*(1-((double)1/numIterUpdate)))+((value)*((double)1/numIterUpdate)));
	
		globalAverage=(globalAverage*(iterations-1)+value)/iterations;
	}

	@Override
	public String toString() {
		return "D: " + deci2.format(dynamicAverage) + " G: " + deci2.format(globalAverage);
	}

	public double getDynamicAverage() {
		return dynamicAverage;
	}

	public double getGlobalAverage() {
		return globalAverage;
	}
}
