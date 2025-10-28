/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Solution;

import Data.Instance;
import Data.Point;

public class Node implements Cloneable
{
	public Node prev,prevOld;
	public Node next,nextOld;
	
	public Route route;
	public int name;
	public int demand;

	public int knn[];
	public int accumulatedDemand=0;
	public boolean nodeBelong;
	public boolean fixedRoute;
	public boolean fixedPosition;
	Point point;
	Instance instance;
	public boolean modified;
	
	public int nameDestinyRoute;
	public int nameOriginRoute;
	public int priority;
	public Route destinyRoute;
	
	public Node(Point point, Instance instance) 
	{
		this.point=point;
		this.instance=instance;
		this.name = point.name;
		this.demand = point.demand;

		this.next=null;
		this.prev=null;
		this.route=null;
		this.knn=instance.getKnn()[name];
	}
	
	public double dist(Node x)
	{
		return instance.dist(x.name,this.name);
	}
	
	public void clean()
	{
		this.nodeBelong=false;
		this.route=null;
	}
	
	 public Node clone() 
	 {
		 Node clone = new Node(point, instance);
		 clone.prev=prev;
		 clone.next=next;
		 clone.route=route;
		 return clone; 
	 }
	
	 public double costRemoval()
	 {
		 return instance.dist(prev.name,next.name)-instance.dist(name,prev.name)-instance.dist(name,next.name);
	 }
	 
	 public double costInsertAfter(Node node)
	 {
		 if(node==null)
			 System.out.println("no null");
		 
		 return -instance.dist(node.name,node.next.name)+instance.dist(name,node.name)+instance.dist(name,node.next.name);
	 }
	 
	@Override
	public String toString() 
	{
		return "|n: "+name+" d: "+demand+"|";
	}

	public int getDemand() {
		return demand;
	}

	public int[] getKnn() {
		return knn;
	}
	
	
}