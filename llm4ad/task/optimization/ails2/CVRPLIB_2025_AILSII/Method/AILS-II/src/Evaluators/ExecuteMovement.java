/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Evaluators;

import Data.Instance;
import Improvement.CandidateNode;
import Solution.Node;
import Solution.Route;

public class ExecuteMovement
{
	Route aRoute, bRoute;
	Node aPrev, bPrev, aNext, bNext;
	Instance instance;

	public ExecuteMovement(Instance instance)
	{
		this.instance = instance;
	}

	public void SWAP(Node a, Node b)
	{
		aRoute = a.route;
		bRoute = b.route;

		if(a.next == b || a.prev == b)
		{
			if(a.next == b)
			{
				aRoute.remove(a);
				aRoute.addAfter(a, b);
			}
			else
			{
				aRoute.remove(b);
				aRoute.addAfter(b, a);
			}
		}
		else
		{
			Node myPrev = a.prev;
			Node nodePrev = b.prev;

			aRoute.remove(a);
			bRoute.remove(b);

			bRoute.addAfter(a, nodePrev);
			aRoute.addAfter(b, myPrev);
		}
	}

	public void SHIFT(Node a, Node b)
	{
		aRoute = a.route;
		bRoute = b.route;

		aRoute.remove(a);
		bRoute.addAfter(a, b);
	}

	public void execute2Opt(Node a, Node b)
	{
		double cost = cost2Opt(a, b);

		aNext = a.next;
		bNext = b.next;
		Node aux = a;
		aux.next = b;
		Node prev = aux.prev;
		while(aux.next != aNext)
		{
			prev = aux;
			aux = aux.next;
			aux.next = aux.prev;
			aux.prev = prev;
		}

		prev = aux;
		aux = aux.next;
		aux.next = bNext;
		aux.prev = prev;

		prev = aux;
		aux = aux.next;
		aux.prev = prev;

		a.route.fRoute += cost;
		a.route.setAccumulatedDemand();
		a.route.modified = true;
	}

	public double cost2Opt(Node a, Node b)
	{
		return -(instance.dist(a.name, a.next.name) + instance.dist(b.name, b.next.name))
		+ (instance.dist(a.name, b.name) + instance.dist(a.next.name, b.next.name));
	}

	public void Cross(Node a, Node b)
	{
		aRoute = a.route;
		bRoute = b.route;

		Node aux = a.next;
		Node prev = b;

		while(aux != aRoute.first)
		{
			aRoute.remove(aux);
			bRoute.addAfter(aux, prev);
			prev = aux;
			aux = a.next;
		}

		Node ultimo = prev;
		aux = ultimo.next;
		prev = a;
		while(aux != bRoute.first)
		{
			bRoute.remove(aux);
			aRoute.addAfter(aux, prev);
			prev = aux;
			aux = ultimo.next;
		}
	}

	public void CrossInverted(Node a, Node b)
	{
		b.route.invertRoute();
		Cross(a, b.prev);
	}

	public double apply(CandidateNode node)
	{
		switch(node.moveType) {
		case SWAP:
			SWAP(node.a, node.b);
			break;
		case SHIFT:
			SHIFT(node.a, node.b);
			break;
		case TwoOpt:
			execute2Opt(node.a, node.b);
			break;
		case Cross:
			Cross(node.a, node.b);
			break;
		case CrossInverted:
			CrossInverted(node.a, node.b);
			break;
		case SWAPStar:
			SHIFT(node.a, node.prevA);
			SHIFT(node.b, node.prevB);
			break;

		}
		return node.cost;
	}
}
