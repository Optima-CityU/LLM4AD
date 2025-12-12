#ifndef GENVRP_ROUTEHISTORYDECOMPOSITION_H
#define GENVRP_ROUTEHISTORYDECOMPOSITION_H

#include "../RouteSequenceDecomposition.h"

namespace genvrp {
    class RouteHistoryDecomposition : public RouteSequenceDecomposition {
        /** Gets a list of subproblems from the given MP individual.
         *
         *  Create subproblems by grouping routes with a k-medoids algorithm. The number of clusters to create
         *  is given by params.data.nbClients / params.deco.targetMaxSpCustomers. The distance between routes
         *  is the inverse of their "historical relatedness". Two routes are close if many of their customers
         *  were visited by the same vehicle in past iterations.
         */
        SPList getSubproblems(const Individual& mpIndividual, const SolverStatus& status) const override;

        std::vector<std::vector<double>> getDistances(const Individual& mpIndividual) const;
        double getRoutesDistance(const std::vector<int>& route1, const std::vector<int>& route2) const;
    public:
        explicit RouteHistoryDecomposition(Params& p) : RouteSequenceDecomposition{p} {}
        ~RouteHistoryDecomposition() = default;
    };
}

#endif //GENVRP_ROUTEHISTORYDECOMPOSITION_H
