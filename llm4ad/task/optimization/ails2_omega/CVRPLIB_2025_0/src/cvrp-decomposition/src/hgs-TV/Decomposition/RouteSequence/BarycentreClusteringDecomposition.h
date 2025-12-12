#ifndef GENVRP_BARYCENTRECLUSTERINGDECOMPOSITION_H
#define GENVRP_BARYCENTRECLUSTERINGDECOMPOSITION_H

#include "../RouteSequenceDecomposition.h"

namespace genvrp {
    class BarycentreClusteringDecomposition : public RouteSequenceDecomposition {
        /** Gets a list of subproblems from the given MP individual.
          *
          * Create subproblems by grouping routes with a k-means algorithm. The number of clusters to create
          * is given by params.data.nbClients / params.deco.targetMaxSpCustomers.
          */
        SPList getSubproblems(const Individual& mpIndividual, const SolverStatus& status) const override;
    public:
        BarycentreClusteringDecomposition(Params& p) : RouteSequenceDecomposition{p} {}
        ~BarycentreClusteringDecomposition() = default;
    };
}

#endif //GENVRP_BARYCENTRECLUSTERINGDECOMPOSITION_H
