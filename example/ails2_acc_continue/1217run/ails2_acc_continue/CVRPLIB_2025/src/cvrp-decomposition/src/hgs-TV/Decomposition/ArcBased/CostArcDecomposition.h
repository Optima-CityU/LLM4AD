#ifndef GENVRP_COSTARCDECOMPOSITION_H
#define GENVRP_COSTARCDECOMPOSITION_H

#include "../ArcBasedDecomposition.h"

namespace genvrp {
    class CostArcDecomposition : public ArcBasedDecomposition {
    protected:
        /** Given a MP individual, gets the shortest 40% of its arcs. */
        std::vector<Arc> getArcsToFix(const Individual& mpIndividual) const override;

    public:
        explicit CostArcDecomposition(Params& p) : ArcBasedDecomposition{p} {}
        ~CostArcDecomposition() override = default;
    };
}

#endif //GENVRP_COSTARCDECOMPOSITION_H
