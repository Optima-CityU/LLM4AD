/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Solution;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import Data.File;
import Data.Instance;
import Data.Point;
import Improvement.IntraLocalSearch;
import SearchMethod.Config;

public class Solution
{
	private Point points[];
	Instance instance;
	Config config;
	protected int size;
	Node solution[];

	protected int first;
	protected Node depot;
	int capacity;
	public Route routes[];
	public int numRoutes;
	protected int numRoutesMin;
	protected int numRoutesMax;
	public double f = 0;
	public int distance;
	double epsilon;
//	-----------Comparadores-----------

	IntraLocalSearch intraLocalSearch;

	public Solution(Instance instance, Config config)
	{
		this.instance = instance;
		this.config = config;
		this.points = instance.getPoints();
		int depot = instance.getDepot();
		this.capacity = instance.getCapacity();
		this.size = instance.getSize() - 1;
		this.solution = new Node[size];
		this.numRoutesMin = instance.getMinNumberRoutes();
		this.numRoutes = numRoutesMin;
		this.numRoutesMax = instance.getMaxNumberRoutes();
		this.depot = new Node(points[depot], instance);
		this.epsilon = config.getEpsilon();

		this.routes = new Route[numRoutesMax];

		for(int i = 0; i < routes.length; i++)
			routes[i] = new Route(instance, config, this.depot, i);

		int count = 0;
		for(int i = 0; i < (solution.length + 1); i++)
		{
			if(i != depot)
			{
				solution[count] = new Node(points[i], instance);
				count++;
			}
		}
	}

	public void clone(Solution reference)
	{
		this.numRoutes = reference.numRoutes;
		this.f = reference.f;

		for(int i = 0; i < routes.length; i++)
		{
			routes[i].nameRoute = i;
			reference.routes[i].nameRoute = i;
		}

		for(int i = 0; i < routes.length; i++)
		{
			routes[i].totalDemand = reference.routes[i].totalDemand;
			routes[i].fRoute = reference.routes[i].fRoute;
			routes[i].numElements = reference.routes[i].numElements;
			routes[i].modified = reference.routes[i].modified;

			if(reference.routes[i].first.prev == null)
				routes[i].first.prev = null;
			else if(reference.routes[i].first.prev.name == 0)
				routes[i].first.prev = routes[i].first;
			else
				routes[i].first.prev = solution[reference.routes[i].first.prev.name - 1];

			if(reference.routes[i].first.next == null)
				routes[i].first.next = null;
			else if(reference.routes[i].first.next.name == 0)
				routes[i].first.next = routes[i].first;
			else
				routes[i].first.next = solution[reference.routes[i].first.next.name - 1];
		}

		for(int i = 0; i < solution.length; i++)
		{
			solution[i].route = routes[reference.solution[i].route.nameRoute];
			solution[i].nodeBelong = reference.solution[i].nodeBelong;

			if(reference.solution[i].prev.name == 0)
				solution[i].prev = routes[reference.solution[i].prev.route.nameRoute].first;
			else
				solution[i].prev = solution[reference.solution[i].prev.name - 1];

			if(reference.solution[i].next.name == 0)
				solution[i].next = routes[reference.solution[i].next.route.nameRoute].first;
			else
				solution[i].next = solution[reference.solution[i].next.name - 1];
		}
	}

	// ------------------------Visualizacao-------------------------

	public String toStringMeu()
	{
		String str = "size: " + size;
		str += "\n" + "depot: " + depot;
		str += "\nnumRoutes: " + numRoutes;
		str += "\ncapacity: " + capacity;

		str += "\nf: " + f;
//		System.out.println(str);
		for(int i = 0; i < numRoutes; i++)
		{
//			System.out.println(str);
			str += "\n" + routes[i];
		}

		return str;
	}

	@Override
	public String toString()
	{
		String str = "";
		for(int i = 0; i < numRoutes; i++)
		{
			str += routes[i].toString2() + "\n";
		}
		str += "Cost " + f + "\n";
		return str;
	}

	public int infeasibility()
	{
		int capViolation = 0;
		for(int i = 0; i < numRoutes; i++)
		{
			if(routes[i].availableCapacity() < 0)
				capViolation += routes[i].availableCapacity();
		}
		return capViolation;
	}

	public boolean checking(String local, boolean feasibility, boolean emptyRoute)
	{
		double f;
		double sumF = 0;
		int sumNumElements = 0;
		boolean erro = false;

		for(int i = 0; i < numRoutes; i++)
		{
			routes[i].findError();
			f = routes[i].F();
			sumF += f;
			sumNumElements += routes[i].numElements;

			if(Math.abs(f - routes[i].fRoute) > epsilon)
			{
				System.out.println("-------------------" + local + " ERROR-------------------" + "\n" + routes[i].toString() + "\nf esperado: " + f);
				erro = true;
			}

			if(emptyRoute && routes[i].first == routes[i].first.next)
			{
				System.out.println("-------------------" + local + " ERROR-------------------" + "Empty route: " + routes[i].toString());
				erro = true;
			}

			if(routes[i].first.name != 0)
			{
				System.out.println("-------------------" + local + " ERROR-------------------" + " Route initiating without depot: " + routes[i].toString());
				erro = true;
			}

			if(feasibility && !routes[i].isFeasible())
			{
				System.out.println("-------------------" + local + " ERROR-------------------" + "Infeasible route: " + routes[i].toString());
				erro = true;
			}

		}
		if(Math.abs(sumF - this.f) > epsilon)
		{
			erro = true;
			System.out.println("-------------------" + local + " Error total sum-------------------");
			System.out.println("Expected: " + sumF + " obtained: " + this.f);
			System.out.println(this.toStringMeu());
		}

		if((sumNumElements - numRoutes) != size)
		{
			erro = true;
			System.out.println("-------------------" + local + " ERROR quantity of Elements-------------------");
			System.out.println("Expected: " + size + " obtained : " + (sumNumElements - numRoutes));

			System.out.println(this);
		}
		return erro;
	}

	public boolean feasible()
	{
		for(int i = 0; i < numRoutes; i++)
		{
			if(routes[i].availableCapacity() < 0)
				return false;
		}
		return true;
	}

	public void removeEmptyRoutes()
	{
		for(int i = 0; i < numRoutes; i++)
		{
			if(routes[i].first == routes[i].first.next)
			{
				removeRoute(i);
				i--;
			}
		}
	}

	private void removeRoute(int index)
	{
		Route aux = routes[index];
		if(index != numRoutes - 1)
		{
			routes[index] = routes[numRoutes - 1];

			routes[numRoutes - 1] = aux;
		}
		numRoutes--;
	}

	public void uploadSolution(String name)
	{
		BufferedReader in;
		try
		{
			in = new BufferedReader(new FileReader(name));
			String str[] = null;
			String line;

			line = in.readLine();
			str = line.split(" ");

			for(int i = 0; i < 3; i++)
				in.readLine();

			int indexRoute = 0;
			line = in.readLine();
			str = line.split(" ");

			System.out.println("-------------- str.length: " + str.length);
			for(int i = 0; i < str.length; i++)
			{
				System.out.print(str[i] + "-");
			}
			System.out.println();

			do
			{
				routes[indexRoute].addNodeEndRoute(depot.clone());
				for(int i = 9; i < str.length - 1; i++)
				{
					System.out.println("add: " + solution[Integer.valueOf(str[i].trim()) - 1] + " na route: " + routes[indexRoute].nameRoute);
					f += routes[indexRoute].addNodeEndRoute(solution[Integer.valueOf(str[i]) - 1]);
				}
				indexRoute++;
				line = in.readLine();
				if(line != null)
					str = line.split(" ");
			}
			while(line != null);

		}
		catch(IOException e)
		{
			System.out.println("File read Error");
		}
	}

	public void uploadSolution1(String name)
	{
		BufferedReader in;
		try
		{
			in = new BufferedReader(new FileReader(name));
			String str[] = null;

			str = in.readLine().split(" ");
			int indexRoute = 0;
			while(!str[0].equals("Cost"))
			{
				for(int i = 2; i < str.length; i++)
				{
					f += routes[indexRoute].addNodeEndRoute(solution[Integer.valueOf(str[i]) - 1]);
				}
				indexRoute++;
				str = in.readLine().split(" ");
			}
		}
		catch(IOException e)
		{
			System.out.println("File read Error");
		}
	}

	public Route[] getRoutes()
	{
		return routes;
	}

	public int getNumRoutes()
	{
		return numRoutes;
	}

	public Node getDepot()
	{
		return depot;
	}

	public Node[] getSolution()
	{
		return solution;
	}

	public int getNumRoutesMax()
	{
		return numRoutesMax;
	}

	public void setNumRoutesMax(int numRoutesMax)
	{
		this.numRoutesMax = numRoutesMax;
	}

	public int getNumRoutesMin()
	{
		return numRoutesMin;
	}

	public void setNumRoutesMin(int numRoutesMin)
	{
		this.numRoutesMin = numRoutesMin;
	}

	public int getSize()
	{
		return size;
	}

	public void printSolution(String end)
	{
		File arq = new File(end);
		arq.write(this.toString());
		arq.close();
	}

	/**
	 * Loads a solution from a file in CVRPLib format.
	 * Format example:
	 */
	/**
	 * Loads a solution from a file in CVRPLib format.
	 * Format example:
	 * Route #1: 1 2 3
	 * Route #2: 4 5 6
	 * Cost 100
	 */
public void loadSolutionCVRPLib(String filePath)
{
    System.out.println("--- Starting loadSolutionCVRPLib ---"); // DEBUG: Start of method
    System.out.println("Reading file: " + filePath); // DEBUG: Verify path

    BufferedReader in;
    try
    {
        in = new BufferedReader(new FileReader(filePath));
        String line;

        // 1. Reset Global Solution State
        this.f = 0;
        this.numRoutes = 0;
        System.out.println("State reset. Cost (f) = 0, numRoutes = 0."); // DEBUG: State reset

        // 2. Clean existing customer nodes
        // This ensures no node thinks it belongs to a previous route
        if (solution != null) {
            int cleanedCount = 0; // DEBUG: Counter
            for (Node n : solution) {
                if (n != null) {
                    n.clean();
                    cleanedCount++;
                }
            }
            System.out.println("Cleaned " + cleanedCount + " existing nodes."); // DEBUG: Cleaning confirmation
        } else {
            System.out.println("Warning: 'solution' array is null. No nodes to clean."); // DEBUG: Null check
        }

        // 3. Re-initialize routes
        // The Route constructor automatically adds the Depot as the start/end point.
        // We recreate them to ensure they are empty of customers.
        System.out.println("Re-initializing " + routes.length + " routes."); // DEBUG: Route init
        for(int i = 0; i < routes.length; i++) {
            routes[i] = new Route(instance, config, this.depot, i);
        }

        int currentRouteIndex = 0;

        while ((line = in.readLine()) != null)
        {
            line = line.trim();

            // Skip empty lines
            if (line.isEmpty()) continue;

            // System.out.println("Processing line: " + line); // DEBUG: (Optional) Very verbose, uncomment if parsing fails completely

            if (line.startsWith("Route"))
            {
                // Example line: "Route #1: 191 309 4 820"

                // Safety check for array bounds
                if (currentRouteIndex >= routes.length) {
                    System.err.println("Error: Solution file has more routes than the instance capacity (" + routes.length + ").");
                    break;
                }

                // 1. Split by ':' to separate "Route #1" from "191 309..."
                String[] parts = line.split(":");
                if(parts.length < 2) {
                    System.out.println("Skipping malformed route line (no colon): " + line); // DEBUG: Malformed line
                    continue;
                }

                // 2. Get the list of IDs string
                String idsPart = parts[1].trim();

                // 3. Split by whitespace to get individual IDs
                // regex \\s+ handles single spaces, multiple spaces, and tabs
                String[] customerIds = idsPart.split("\\s+");

//                 System.out.println("Loading Route index " + currentRouteIndex + " with " + customerIds.length + " potential tokens."); // DEBUG: Route details

                for (String idStr : customerIds)
                {
                    if(idStr.isEmpty()) continue;

                    try {
                        int id = Integer.parseInt(idStr);

                        // CVRPLib IDs are usually 1-based (1 is the first customer).
                        // The solution[] array is 0-based.
                        // solution[0] is Customer 1.
                        int arrayIndex = id - 1;

                        if (arrayIndex >= 0 && arrayIndex < solution.length) {
                            Node customerNode = solution[arrayIndex];

                            // addNodeEndRoute returns the cost increase (delta)
                            // It handles linking prev/next and setting the route inside the Node object
                            double costIncrease = routes[currentRouteIndex].addNodeEndRoute(customerNode);

                            this.f += costIncrease;
                            // System.out.println("  -> Added Node ID " + id + " (Idx: " + arrayIndex + "). Cost delta: " + costIncrease); // DEBUG: (Optional) Verbose node addition
                        } else {
                            System.err.println("Warning: Node ID " + id + " is out of bounds for solution array (Length: " + solution.length + ").");
                        }
                    } catch (NumberFormatException nfe) {
                        System.out.println("  -> Ignored non-integer token: '" + idStr + "'"); // DEBUG: Parsing check
                    }
                }

//                 System.out.println("Finished Route index " + currentRouteIndex + ". Current Total Cost: " + this.f); // DEBUG: Route finished

                // Move to the next route slot
                currentRouteIndex++;
            }
            else if (line.startsWith("Cost"))
            {
                // Optional: Read the cost from file for validation
                try {
                    String[] parts = line.split(" ");
                    if (parts.length > 1) {
                        double fileCost = Double.parseDouble(parts[1]);
//                         System.out.println("--- Validation ---"); // DEBUG: Validation header
//                         System.out.println("File Cost: " + fileCost); // DEBUG: File cost
//                         System.out.println("Calc Cost: " + this.f);   // DEBUG: Calculated cost
                        if (Math.abs(fileCost - this.f) > 0.01) {
                            System.err.println("WARNING: Calculated cost differs significantly from file cost!");
                        }
                    }
                } catch (Exception e) {
                    System.out.println("Could not parse Cost line for validation.");
                }
            }
        }

        // Update the actual number of routes used
        this.numRoutes = currentRouteIndex;
//         System.out.println("Total loaded routes: " + this.numRoutes); // DEBUG: Final route count
//         System.out.println("Final Calculated Cost: " + this.f); // DEBUG: Final cost
//         System.out.println("--- End loadSolutionCVRPLib ---"); // DEBUG: End of method

        in.close();
    }
    catch (IOException e)
    {
        System.out.println("Error reading solution file: " + e.getMessage());
        e.printStackTrace();
    }
}



}
