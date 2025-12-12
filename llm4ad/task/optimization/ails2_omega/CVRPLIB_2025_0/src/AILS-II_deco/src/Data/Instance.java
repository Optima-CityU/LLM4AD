/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import SearchMethod.Config;
import SearchMethod.InputParameters;

public class Instance 
{
	private int size;
	private double averagePerRoute;
	private int knn[][];
	private float dist[][];
	private NodeKnn[]neighKnn;
	private int capacity;
	private int depot;
	private int minNumberRoutes;
	private int maxNumberRoutes;
	private Point points[];
	double coord[][];
	private String name;
	private double largestDist=0;
	private double minDist=Integer.MAX_VALUE;
	EdgeType edgeType;
	double distance;
	boolean rounded;
	double sumDemand;
	String str[];
	Config config;
	boolean print = false;
	
	public Instance(InputParameters reader) 
	{
		this.name=reader.getFile();
		this.config=reader.getConfig();
		this.rounded=reader.isRounded();
		
		BufferedReader in;
		try 
		{
			in = new BufferedReader(new FileReader(name));
			
			str=in.readLine().split(":");
			while(!str[0].trim().equals("EOF"))
			{
				switch(str[0].trim())
				{
					case "DIMENSION": size=Integer.valueOf(str[1].trim());break;
					case "EDGE_WEIGHT_TYPE":
					case "EDGE_WEIGHT_FORMAT": setCoordType(str[1].trim());break;
					case "CAPACITY": capacity=Integer.valueOf(str[1].trim());break;
					case "NODE_COORD_SECTION": readCoord(in);break;
					case "EDGE_WEIGHT_SECTION": readDistMatrix(in);break;
					case "DEMAND_SECTION": readDemand(in);break;
					case "DEPOT_SECTION": readDepot(in);break;
				}
				str=in.readLine().split(":");
			}
				
			minNumberRoutes=(int) Math.ceil(sumDemand/capacity);
			averagePerRoute=(double)size/minNumberRoutes;
			double percent=minNumberRoutes*((double)1/averagePerRoute);
			maxNumberRoutes=minNumberRoutes+(3+(int)Math.ceil(percent));
			
		} 
		catch (IOException e) {
	    	System.out.println("File reading error");
	    }
		neighKnn=null;
	}

	public double dist(int i,int j)
	{
		return dist[i][j];
	}
	
	public double distCalc(int i,int j)
	{
		if(i!=j)
		{
			if(edgeType==EdgeType.EUC_2D)
			{
				if(rounded)
					return roundedDistance(points[i].x,points[i].y,points[j].x,points[j].y);
				else
					return distance(points[i].x,points[i].y,points[j].x,points[j].y);
			}
			else
				return distanceGeo(coord[i][0],coord[i][1],coord[j][0],coord[j][1]);
		}
		else
			return 0;
	}
	
	private double distance(double x1,double y1,double x2,double y2)
	{
		return Math.sqrt(((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2)));
	}
	
	private int roundedDistance(double x1,double y1,double x2,double y2)
	{
		return (short) Math.round(Math.sqrt(((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2))));
	}
	
	private void setCoordType(String str)
	{
		if(str.equals("EUC_2D"))
			edgeType=EdgeType.EUC_2D;
		else if(str.equals("EXPLICIT"))
			edgeType=EdgeType.EXPLICIT;
		else
		{
			edgeType=EdgeType.Coord;
			coord=new double[size][2];
		}
	}
	
	private void readCoord(BufferedReader in)
	{
		if(print)
			System.out.println("readCoord");
		
		points=new Point [size];
		for (int i = 0; i < points.length; i++) 
			points[i]=new Point(i);
		
		try 
		{
			for (int i = 0; i < size; i++) 
			{
				if(edgeType==EdgeType.EUC_2D)
				{
					str=in.readLine().trim().replaceAll("\\s+"," ").split("[' '|'\t']");
					points[i].x=Double.valueOf(str[1].trim());
					points[i].y=Double.valueOf(str[2].trim());
				}
				else
				{
					str=in.readLine().split("[' '|'\t']");
					coord[i][0]=Double.valueOf(str[1].trim());
					coord[i][1]=Double.valueOf(str[2].trim());
					points[i].x=Double.valueOf((int) (coord[i][0]*10000));
					points[i].y=Double.valueOf((int) (coord[i][1]*10000));
				}
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		if(edgeType==EdgeType.EUC_2D)
		{
			int knnLimit=Math.min(config.getKnnLimit(), size-1);
			knn=new int[size][knnLimit];
			neighKnn=new NodeKnn[size-1];
			
			for (int i = 0; i < neighKnn.length; i++) 
				neighKnn[i]=new NodeKnn();
			
			dist=new float[size][size];
			
			int counter=0;
			for (int i = 0; i < size; i++) 
			{
				counter=0;
				for (int j = 0; j < size; j++) 
				{
					distance=distCalc(i,j);
					
					dist[i][j]=(float)distance;
					
					if(i<j)
					{
						if(largestDist<distance)
							largestDist=distance;
						
						if(minDist>distance)
							minDist=distance;
					}
					
					if(i!=j)
					{
						neighKnn[counter].dist=distance;
						neighKnn[counter].name=j;
						counter++;
					}
				}
				
				Arrays.sort(neighKnn);
				
				for (int j = 0; j < knnLimit; j++) 
					knn[i][j]=neighKnn[j].name;
			}
		}
	}
	
	private void readDistMatrix(BufferedReader in)
	{
		int limitKnn=Math.min(config.getKnnLimit(), size-1);
		knn=new int[size][limitKnn];
		neighKnn=new NodeKnn[size-1];
		
		for (int i = 0; i < neighKnn.length; i++) 
			neighKnn[i]=new NodeKnn();
		
		dist=new float[size][size];
				
		try 
		{
			for (int i = 1; i < size; i++) 
			{
				str=in.readLine().trim().replaceAll("\\s+"," ").split("[' '|'\t']");
				for (int j = 0; j < str.length; j++) 
				{
					distance=Double.valueOf(str[j].trim());
					dist[i][j]=(short) distance;
					dist[j][i]=dist[i][j];
				}
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		int count=0;
		for (int i = 0; i < size; i++) 
		{
			count=0;
			for (int j = 0; j < size; j++) 
			{
				distance=dist[i][j];
				
				if(i<j)
				{
					if(largestDist<distance)
						largestDist=distance;
					
					if(minDist>distance)
						minDist=distance;
				}
				
				if(i!=j)
				{
					neighKnn[count].dist=distance;
					neighKnn[count].name=j;
					count++;
				}
			}
			
			Arrays.sort(neighKnn);
			
			for (int j = 0; j < limitKnn; j++) 
				knn[i][j]=neighKnn[j].name;
			
		}
	}
	
	private void readDemand(BufferedReader in)
	{
		if(print)
			System.out.println("readDemand");
		
		try {
			sumDemand=0;
			for (int i = 0; i < size; i++) 
			{
				str=in.readLine().trim().replaceAll("\\s+"," ").split("[' '|'\t']");
				points[i].demand=Integer.valueOf(str[1]);
				sumDemand+=points[i].demand;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void readDepot(BufferedReader in)
	{
		try {
				depot=Integer.valueOf(in.readLine().trim())-1;
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private int distanceGeo(double Lat1,double Long1,double Lat2,double Long2)
	{
		double firstLatToRad = Math.toRadians(Lat1);
		double secondLatToRad = Math.toRadians(Lat2);

		double deltaLongitudeInRad = Math.toRadians(Long2
		- Long1);

		return (int) Math.round(Math.acos(Math.cos(firstLatToRad) * Math.cos(secondLatToRad)
		* Math.cos(deltaLongitudeInRad) + Math.sin(firstLatToRad)
		* Math.sin(secondLatToRad))
		* 6378100);
	}
	
	@Override
	public String toString() {
		return "size=" + size + "\n capacity=" + capacity
				+ "\ndepot="+ depot +" minNumberRoutes: "+minNumberRoutes;
	}

	public Point[] getPoints() {
		return points;
	}

	public void setPoints(Point[] points) {
		this.points = points;
	}

	public int getMinNumberRoutes() {
		return minNumberRoutes;
	}

	public int getMaxNumberRoutes() {
		return maxNumberRoutes;
	}

	public int getSize() {
		return size;
	}

	public void setSize(int size) {
		this.size = size;
	}

	public int getCapacity() {
		return capacity;
	}

	public void setCapacity(int capacity) {
		this.capacity = capacity;
	}

	public int getDepot() {
		return depot;
	}

	public void setDepot(int depot) {
		this.depot = depot;
	}

	public double getLargestDist() {
		return largestDist;
	}
	
	public double getMinDist() {
		return minDist;
	}

	public String getName() {
		return name;
	}

	public int[][] getKnn() {
		return knn;
	}
	
}
