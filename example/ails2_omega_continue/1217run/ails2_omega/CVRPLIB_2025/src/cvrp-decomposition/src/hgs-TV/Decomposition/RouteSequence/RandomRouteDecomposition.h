#ifndef GENVRP_RANDOMROUTEDECOMPOSITION_H
#define GENVRP_RANDOMROUTEDECOMPOSITION_H

#include "../../Individual.h"
#include "../../SolverStatus.h"
#include "../RouteSequenceDecomposition.h"

namespace genvrp {
    class RandomRouteDecomposition : public RouteSequenceDecomposition {
        /** Gets a list of subproblems from the given MP individual.
         *
         *  The subproblems are obtained by shuffling the routes and scanning them in the given randomised order.
         *  Customers are accumulated while the open subproblem size is below a threshold; after that, the subproblem
         *  is "closed" and a new one is open. This is repeated until all routes are scanned.
         */
        SPList getSubproblems(const Individual& mpIndividual, const SolverStatus& status) const override;
    public:
        explicit RandomRouteDecomposition(Params& p) : RouteSequenceDecomposition{p} {}
        ~RandomRouteDecomposition() = default;
    };
}

#endif //GENVRP_RANDOMROUTEDECOMPOSITION_H
