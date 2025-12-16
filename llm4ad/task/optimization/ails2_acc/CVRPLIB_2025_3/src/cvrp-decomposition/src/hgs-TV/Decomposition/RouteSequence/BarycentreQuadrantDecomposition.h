#ifndef GENVRP_BARYCENTREQUADRANTDECOMPOSITION_H
#define GENVRP_BARYCENTREQUADRANTDECOMPOSITION_H

#include "../RouteSequenceDecomposition.h"

namespace genvrp {
    class BarycentreQuadrantDecomposition : public RouteSequenceDecomposition {
        /** Gets a list of subproblems from the given MP individual.
         *
         *  The method generates at most four subproblems, one for each quadrant. Routes having their
         *  barycentre in the same quadrant are grouped together.
         */
        SPList getSubproblems(const Individual& mpIndividual, const SolverStatus& status) const override;

    public:
        explicit BarycentreQuadrantDecomposition(Params& p) : RouteSequenceDecomposition{p} {}
        ~BarycentreQuadrantDecomposition() = default;
    };
}

#endif //GENVRP_BARYCENTREQUADRANTDECOMPOSITION_H
