/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Data;

public class InstanceData
{
	public String name;
	public Best bestSolution;
	public Best bestSavedSolution;
	public boolean optimal;
	public boolean savedSolutionFlag;
	public boolean rounded;

	public InstanceData(String name, double bestSolution, boolean rounded, boolean optimal, int bestSavedSolution, boolean savedSolutionFlag)
	{
		super();
		this.name = name;
		this.bestSolution = new Best(bestSolution);
		this.rounded = rounded;
		this.optimal = optimal;
		this.savedSolutionFlag = savedSolutionFlag;
		this.bestSavedSolution = new Best(bestSavedSolution);
	}

	public InstanceData(String name, double bestSolution, boolean rounded, boolean optimal, boolean savedSolutionFlag)
	{
		super();
		this.name = name;
		this.bestSolution = new Best(bestSolution);
		this.bestSavedSolution = new Best(bestSolution);
		this.rounded = rounded;
		this.optimal = optimal;
		this.savedSolutionFlag = savedSolutionFlag;
	}

	@Override
	public String toString()
	{
		return "name: " + name + "\tbestSolution: " + bestSolution.getOptimal() + "\toptimal: " + optimal + "\tsavedSolutionFlag: " + savedSolutionFlag;
	}

	public String getName()
	{
		return name;
	}

	public void setName(String name)
	{
		this.name = name;
	}

	public Best getBestSolution()
	{
		return bestSolution;
	}

	public void setBestSolution(Best bestSolution)
	{
		this.bestSolution = bestSolution;
	}

	public boolean isOptimal()
	{
		return optimal;
	}

	public void setOptimal(boolean optimal)
	{
		this.optimal = optimal;
	}

	public boolean isSavedSolutionFlag()
	{
		return savedSolutionFlag;
	}

	public void setSavedSolutionFlag(boolean savedSolutionFlag)
	{
		this.savedSolutionFlag = savedSolutionFlag;
	}

	public Best getBestSavedSolution()
	{
		return bestSavedSolution;
	}

	public void setBestSavedSolution(Best bestSavedSolution)
	{
		this.bestSavedSolution = bestSavedSolution;
	}

}
