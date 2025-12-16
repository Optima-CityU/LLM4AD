/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Solution;

import Data.Instance;
import SearchMethod.Config;

public class Route implements Comparable<Route>
{
	public int  capacity;
	public int depot;

	public Node first;
	public boolean modified;
	public int totalDemand=0;
	public double fRoute=0,prevF;
	double increaseObjFunction;
	public int numElements=0;
	public int nameRoute;
	Node prev,next;
	
	//busca local best
	public double lowestCost=0;
	Node aux;
	
	//------------------------------------------------
	public double cost,costRemoval;
	public Node bestCost;
	public double distance,lowestDist;
	//------------------------------------------------
	public boolean update;
	
    Instance instance;
    int limitAdj;
    
	public Route(Instance instance, Config config, Node depot,int nameRoute) 
	{
		this.instance=instance;
		this.capacity = instance.getCapacity();
		this.depot=depot.name;
		this.nameRoute=nameRoute;
		this.first=null;
		this.limitAdj=config.getVarphi();
		addNodeEndRoute(depot.clone());
	}
	
    public Node findBestPosition(Node no)
	{
		bestCost=first;
		aux=first.next;
		lowestCost=instance.dist(first.name,no.name)+instance.dist(no.name,first.next.name)-instance.dist(first.name,first.next.name);
		do
		{
			cost=instance.dist(aux.name,no.name)+instance.dist(no.name,aux.next.name)-instance.dist(aux.name,aux.next.name);
			if(cost<lowestCost)
			{
				lowestCost=cost;
				bestCost=aux;
			}
			aux=aux.next;
		}
		while(aux!=first);
		return bestCost;
	}
    
    public Node findBestPositionExceptAfterNode(Node no,Node exception)
  	{
  		bestCost=first;
  		aux=first.next;
  		
  		lowestCost=instance.dist(first.name,no.name)+instance.dist(no.name,first.next.name)-instance.dist(first.name,first.next.name);
  		do
  		{
  			cost=instance.dist(aux.name,no.name)+instance.dist(no.name,aux.next.name)-instance.dist(aux.name,aux.next.name);
  			if(cost<lowestCost&&aux.name!=exception.name)
  			{
  				lowestCost=cost;
  				bestCost=aux;
  			}
  			aux=aux.next;
  		}
  		while(aux!=first);
  		return bestCost;
  	}
    
    public Node findBestPositionExceptAfterNodeKNN(Node node,Node exception, Node solution[])
   	{
    	lowestCost=Double.MAX_VALUE;
    	
   		bestCost=first;
   		
   		if(numElements>limitAdj)
   		{
   			for (int j = 0; j < limitAdj; j++) 
   			{
   	   			if(node.getKnn()[j]==0)
   	   			{
   	   				aux=first;
   	   				cost=instance.dist(aux.name,node.name)+instance.dist(node.name,aux.next.name)-instance.dist(aux.name,aux.next.name);
   	   				if(cost<lowestCost)
   	   				{
   	   					lowestCost=cost;
   	   					bestCost=aux;
   	   				}   				
   	   			}
   	   			else if(node.getKnn()[j]!=exception.name&&solution[node.getKnn()[j]-1].route==this)
   	   			{
   	   				aux=solution[node.getKnn()[j]-1];
   	   				cost=instance.dist(aux.name,node.name)+instance.dist(node.name,aux.next.name)-instance.dist(aux.name,aux.next.name);
   	   				if(cost<lowestCost)
   	   				{
   	   					lowestCost=cost;
   	   					bestCost=aux;
   	   				}
   	   			}
   	   			
   			}
   		}
   		else
   		{
   			findBestPositionExceptAfterNode(node,exception);
   		}
   		
   		return bestCost;
   	}
    
	public void clean()
	{
		first.prev=first;
		first.next=first;
		
		totalDemand=0;
		fRoute=0;
		numElements=1;
		modified=true;
	}
	
	
	public void setAccumulatedDemand()
	{
		first.accumulatedDemand=0;
		aux=first.next;
		do
		{
			aux.accumulatedDemand=aux.prev.accumulatedDemand+aux.demand;
			aux=aux.next;
		}
		while(aux!=first);
	}
	
	public void invertRoute()
	{
		aux=first;
		Node prev=first.prev;
		Node next=first.next;
		do
		{
			aux.prev=next;
			aux.next=prev;
			aux=next;
			prev=aux.prev;
			next=aux.next;
		}
		while(aux!=first);
	}
	
	public double F()
	{
		double f=0;
		Node node=first;
		Node next=node.next;
		do
		{
			f+=instance.dist(node.name,next.name);
			node=next;
			next=node.next;
		}
		while(next!=first);
		
		f+=instance.dist(node.name,next.name);
		return f;
	}
	
	public void findError()
	{
		int count=0;
		Node aux=first;
		
		do
		{
			if(aux.prev==null||aux.next==null)
			{
				System.out.println("Erro no null No: "+aux+" em:\n"+this.toString());
				System.out.println(this);
			}
			if(aux.route!=this)
				System.out.println("Erro no : "+aux+" com route Errada:\n"+this.toString());

			count++;
			aux=aux.next;
		}
		while(aux!=first);
		if(count!=numElements)
			System.out.println("Error in numElements \n"+this.toString());

	}
	
	//------------------------Visualizacao-------------------------

	@Override
	public String toString() 
	{
		String str="Route: "+nameRoute;
		str+=" f: "+fRoute;
		str+=" space: "+availableCapacity();
		str+=" size: "+numElements+" = ";
		str+=" modified: "+modified+" = ";

		Node aux=first;
		do
		{
			str+=aux+"->";
//			System.out.println(str);
			aux=aux.next;
		}
		while(aux!=first);
		
		return str;
	}
	
	public String toString2() 
	{
		String str="Route #"+(nameRoute+1)+": ";
		Node aux=first.next;
		do
		{
			str+=aux.name+" ";
			aux=aux.next;
		}
		while(aux!=first);
		
		return str;
	}
	
	public boolean isFeasible()
	{
		if((capacity-totalDemand)>=0)
			return true;
		return false;
	}
	
	public int availableCapacity()
	{
		return capacity-totalDemand;
	}
	
	public void setIncreaseObjFunction(int increaseObjFunction) {
		this.increaseObjFunction = increaseObjFunction;
	}

	public int getNumElements() {
		return numElements;
	}

	public void setNumElements(int numElements) {
		this.numElements = numElements;
	}
	
	@Override
	public int compareTo(Route x) 
	{
		return nameRoute-x.nameRoute;
	}

	public double remove(Node node) 
	{
		double cost=node.costRemoval();
		
		if(node==first)
			first=node.prev;
		
		modified=true;
		node.modified=true;
		node.next.modified=true;
		node.prev.modified=true;
		
		node.nodeBelong=false;
		
		fRoute+=cost;
		numElements--;
		
		prev=node.prev;
		next=node.next;
		
		prev.next=next;
		next.prev=prev;
		
		totalDemand-=node.demand;
		
		node.route=null;
		node.prev=null;
		node.next=null;
		
		return cost;
	}

	//------------------------Add No-------------------------
	
	public double addNodeEndRoute(Node node)
	{
		node.route=this;
		if(first==null)
		{
			first=node;
			first.prev=node;
			first.next=node;
			numElements++;
			totalDemand=0;
			fRoute=0;
			return 0;
		}
		else if(node.name==0)
		{
			aux=first.prev;
			first=node;
			return addAfter(node, aux); 
		}
		else 
		{
			aux=first.prev;
			return addAfter(node, aux); 
		}
	}
	
	public double addAfter(Node node1, Node node2) 
	{
		increaseObjFunction=node1.costInsertAfter(node2);
		modified=true;
		node1.nodeBelong=true;
		node1.modified=true;
		node2.next.modified=true;
		node2.prev.modified=true;
		
		numElements++;
		fRoute+=increaseObjFunction;
		
		totalDemand+=node1.demand;

		node1.route=this;
		
		next=node2.next;
		node2.next=node1;
		node1.prev=node2;
		
		next.prev=node1;
		node1.next=next;
		
		if(node1.name==0)
			first=node1;
		
		return increaseObjFunction;
	}
}

