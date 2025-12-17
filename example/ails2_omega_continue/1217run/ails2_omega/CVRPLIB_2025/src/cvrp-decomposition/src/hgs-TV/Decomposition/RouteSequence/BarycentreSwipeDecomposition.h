#ifndef GENVRP_BARYCENTRESWIPEDECOMPOSITION_H
#define GENVRP_BARYCENTRESWIPEDECOMPOSITION_H

#include "../RouteSequenceDecomposition.h"

namespace genvrp {
    class BarycentreSwipeDecomposition : public RouteSequenceDecomposition {
        /** Gets a list of subproblems from the given MP individual.
         *
         *  The subproblems are obtained by sorting the routes by the angle of their barycentres and scanning them
         *  in the given order, starting from a random position. Customers are accumulated while the open subproblem
         *  size is below a threshold;\after that, the subproblem is "closed" and a new one is open. This is repeated
         *  until all routes are scanned.
         */
        SPList getSubproblems(const Individual& mpIndividual, const SolverStatus& status) const override;

    public:
        explicit BarycentreSwipeDecomposition(Params& p) : RouteSequenceDecomposition{p} {}
        ~BarycentreSwipeDecomposition() = default;
    };
}

#endif //GENVRP_BARYCENTRESWIPEDECOMPOSITION_H
