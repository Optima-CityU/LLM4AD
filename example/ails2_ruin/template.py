import os

template_program = '''
package Perturbation;

import java.util.HashMap;

import Data.Instance;
import DiversityControl.OmegaAdjustment;
import Improvement.IntraLocalSearch;
import SearchMethod.Config;
import Solution.Node;
import Solution.Solution;

//Ruinnew
public class Ruinnew extends Perturbation
{
	public Ruinnew(Instance instance, Config config,
	HashMap<String, OmegaAdjustment> omegaSetup, IntraLocalSearch intraLocalSearch)
	{
		super(instance, config, omegaSetup,intraLocalSearch);
		this.perturbationType=PerturbationType.Ruinnew;
	}

	public void applyPerturbation(Solution s)
	{
		setSolution(s);
		
//		---------------------------------------------------------------------
//		only a placeholder, replace with your design

		Node reference=solution[rand.nextInt(solution.length)];
		for (int i = 0; i < omega&&i < reference.knn.length&&countCandidates<omega; i++) 
		{
			if(reference.knn[i]!=0)
			{
				node=solution[reference.knn[i]-1];
				candidates[countCandidates]=node;
				countCandidates++;
				node.prevOld=node.prev;
				node.nextOld=node.next;
				f+=node.route.remove(node);
			}
		}
		
		setOrder();
		
		addCandidates();
		
		assignSolution(s);
	}
}

---

### Core Class Summaries for Your Reference

`Instance` Class Summary:
This class holds the problem data.
   `dist(int i, int j)`: Returns the distance between node `i` and node `j`.
   `getKnn()`: Returns a 2D integer array `int[][]`. `getKnn()[i]` is an array of the nearest neighbor indices for node `i`.

`Node` Class Summary:
This class represents a customer or the depot.
   `int name`: The unique identifier of the node (1 to N for customers, 0 for depot).
   `int[] knn`: An array containing the names (IDs) of the k-nearest neighbors to this node.
   `Route route`: The route this node currently belongs to.
   `Node prev`: The previous node in the route.
   `Node next`: The next node in the route.
   `boolean nodeBelong`: A flag that is `true` if the node is part of a route, `false` otherwise.
   `double costRemoval()`: Calculates the change in cost if this node is removed from its route. The formula is `dist(prev, next) - dist(prev, this) - dist(this, next)`.

`Route` Class Summary:
This class represents a single vehicle route.
   `double remove(Node node)`: Removes the specified `node` from this route.
       It correctly updates the `prev` and `next` pointers of the surrounding nodes.
       It sets the `node.nodeBelong` flag to `false`.
       It updates the route's total cost (`fRoute`) and other properties.
       Crucially, it returns the change in objective function value (the cost savings) from the removal.** This return value should be added to the total solution cost `f`.
   `double addAfter(Node nodeToInsert, Node positionNode)`: Inserts `nodeToInsert` immediately after `positionNode` in the route. It returns the change in cost.

`Perturbation` (Parent Class) Summary:
The `Ruinnew` class extends `Perturbation` and inherits these important members:
   `Node[] solution`: An array of all customer nodes in the problem, indexed from 0 to N-1. You can access a node with ID `i` via `solution[i-1]`.
   `double f`: The total cost (objective function value) of the current solution.
   `double omega`: The target number of customers to remove.
   `Node[] candidates`: An array to store the nodes that have been removed.
   `int countCandidates`: The current number of nodes in the `candidates` array.
   `Random rand`: A random number generator.
'''

task_description = """
Write a new ruin operator in java for solving CVRP.
"""

java_dir = "CVRPLIB_2025"        # 多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
aim_java_relative_path = os.path.join('src', 'AILS-II_origin','src', 'Perturbation', 'Ruinnew.java')       # 被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"

# java_dir = "CVRPLIB_2025_AILSII"        # 多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
# aim_java_relative_path = os.path.join('Method', 'AILS-II','src', 'DiversityControl', 'DistAdjustment.java')       # 被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"



