/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package SearchMethod;

import java.text.DecimalFormat;
import java.util.Arrays;

import Perturbation.InsertionHeuristic;
import Perturbation.PerturbationType;

public class Config implements Cloneable
{
	DecimalFormat deci=new DecimalFormat("0.000");
	double etaMin,etaMax;
	int dMin,dMax;	
	
	int gamma;
	PerturbationType perturbation[];
	InsertionHeuristic[]insertionHeuristics;
	//--------------------PR-------------------
	int varphi;
	double epsilon;
	int knnLimit;
	StoppingCriterionType stoppingCriterionType;
	
	public Config() 
	{
//		----------------------------Main----------------------------
		this.stoppingCriterionType=StoppingCriterionType.Time;
		this.dMin=15;
		this.dMax=30;
		this.gamma=30; 
		this.knnLimit=100;
		this.varphi=40;
		
		
		this.epsilon=0.01;
		this.etaMin=0.01;
		this.etaMax=1;
		
		this.perturbation=new PerturbationType[2];
		this.perturbation[0]=PerturbationType.Sequential;
		this.perturbation[1]=PerturbationType.Concentric;
		
		this.insertionHeuristics=new InsertionHeuristic[2];
		insertionHeuristics[0]=InsertionHeuristic.Distance;
		insertionHeuristics[1]=InsertionHeuristic.Cost;
	}
	
	public Config clone()
	{
		try {
			return (Config) super.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	@Override
	public String toString() 
	{
		return "Config "
		+"\nstoppingCriterionType: "+stoppingCriterionType
		+"\netaMax: "+deci.format(etaMax)
		+"\netaMin: "+deci.format(etaMin)
		+"\ngamma: "+gamma
		+"\ndMin: "+dMin
		+"\ndMax: "+dMax
		+"\nvarphi: "+varphi
		+"\nepsilon: " + deci.format(epsilon)
		+"\nperturbation: "+Arrays.toString(perturbation)
		+"\ninsertionHeuristics: "+Arrays.toString(insertionHeuristics)
		+"\nlimitKnn: "+knnLimit
		;  
	}
	public DecimalFormat getDeci() {
		return deci;
	}

	public void setDeci(DecimalFormat deci) {
		this.deci = deci;
	}

	public double getEtaMin() {
		return etaMin;
	}

	public void setEtaMin(double etaMin) {
		this.etaMin = etaMin;
	}

	public double getEtaMax() {
		return etaMax;
	}

	public void setEtaMax(double etaMax) {
		this.etaMax = etaMax;
	}

	public int getDMin() {
		return dMin;
	}

	public void setDMin(int dMin) {
		this.dMin = dMin;
	}

	public int getDMax() {
		return dMax;
	}

	public void setDMax(int dMax) {
		this.dMax = dMax;
	}

	public int getGamma() {
		return gamma;
	}

	public void setGamma(int gamma) {
		this.gamma = gamma;
	}

	public PerturbationType[] getPerturbation() {
		return perturbation;
	}

	public void setPerturbation(PerturbationType[] perturbation) {
		this.perturbation = perturbation;
	}

	public InsertionHeuristic[] getInsertionHeuristics() {
		return insertionHeuristics;
	}

	public void setInsertionHeuristics(InsertionHeuristic[] insertionHeuristics) {
		this.insertionHeuristics = insertionHeuristics;
	}

	public int getVarphi() {
		return varphi;
	}

	public void setVarphi(int varphi)
	{
		if(knnLimit<varphi)
			knnLimit=varphi;
		
		this.varphi = varphi;
	}

	public double getEpsilon() {
		return epsilon;
	}

	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}

	public int getKnnLimit() {
		return knnLimit;
	}

	public void setKnnLimit(int knnLimit) {
		this.knnLimit = knnLimit;
	}

	public StoppingCriterionType getStoppingCriterionType() {
		return stoppingCriterionType;
	}

	public void setStoppingCriterionType(StoppingCriterionType stoppingCriterionType) {
		this.stoppingCriterionType = stoppingCriterionType;
	}


}
