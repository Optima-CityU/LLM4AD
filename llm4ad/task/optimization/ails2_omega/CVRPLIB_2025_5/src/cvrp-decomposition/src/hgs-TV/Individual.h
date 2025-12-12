#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "Params.h"

namespace genvrp {

    class Individual;

    struct CostSol {
        double penalizedCost;  // Penalized cost of the solution
        int nbRoutes;          // Number of routes
        double distance;       // Total Distance
        double capacityExcess; // Sum of excess load in all routes
        double durationExcess; // Sum of excess duration in all routes
        CostSol() {
            penalizedCost = 0.;
            nbRoutes = 0;
            distance = 0.;
            capacityExcess = 0.;
            durationExcess = 0.;
        }
    };

    /** Structure used to keep information about the barycentre of a route. */
    struct Barycentre {
        /** X coordinate (average of the x coordinates of the visited customers). */
        double x;

        /** Y coordinate (average of the y coordinates of the visited customers). */
        double y;

        /** Polar angle of the barycentre. */
        double angle;

        /** Simple constructor assigning all members. */
        Barycentre(double x, double y, double angle) : x{x}, y{y}, angle{angle} {}
    };

    class Individual {
    public:
        Params* params;                       // Problem parameters
        CostSol cost;                         // Solution cost parameters
        std::vector<int> giantTour;           // Giant tour representing the individual
        std::vector<std::vector<int>> routes; // For each vehicle, the associated sequence of deliveries (complete solution)
        std::vector<Barycentre> barycentres;  // Barycentres of each route. If a route is empty, the barycentre is meaningless and is at (0,0)
        std::vector<int> successors;          // For each node, the successor in the solution (can be the depot 0)
        std::vector<int> predecessors;        // For each node, the predecessor in the solution (can be the depot 0)
        std::multiset<std::pair<double, Individual*>> indivsPerProximity; // The other individuals in the population, ordered by increasing proximity (the set
                                                                          // container follows a natural ordering based on the first value of the pair)
        bool isFeasible;                                                  // Feasibility status of the individual
        double biasedFitness;                                             // Biased fitness of the solution

        // Measuring cost of a solution from the information of routes
        void evaluateCompleteCost();

        // Removing an individual in the structure of proximity
        void removeProximity(Individual* indiv);

        // Distance measure with another individual
        double brokenPairsDistance(Individual* indiv2);

        // Returns the average distance of this individual with the nbClosest individuals
        double averageBrokenPairsDistanceClosest(int nbClosest);

        // Exports a solution in CVRPLib format (adds a final line with the computational time)
        void exportCVRPLibFormat(std::string fileName);

        // Reads a solution in CVRPLib format, returns TRUE if the process worked, or FALSE if the file does not exist or is not readable
        static bool readCVRPLibFormat(std::string fileName, std::vector<std::vector<int>>& readSolution, double& readCost);

        // Constructor: random individual
        Individual(Params* params);

        // Constructor: empty individual
        Individual();
    };

} // namespace genvrp

#endif
