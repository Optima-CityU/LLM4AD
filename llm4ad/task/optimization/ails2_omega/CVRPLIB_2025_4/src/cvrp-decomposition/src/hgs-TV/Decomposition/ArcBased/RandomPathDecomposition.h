#ifndef GENVRP_RANDOMPATHDECOMPOSITION_H
#define GENVRP_RANDOMPATHDECOMPOSITION_H

#include "../ArcBasedDecomposition.h"
#include "../../Params.h"

#include <vector>

namespace genvrp {
    class RandomPathDecomposition : public ArcBasedDecomposition {
    protected:
        /** Given a MP individual, gets some of its paths (disjoint) at random. */
        std::vector<Arc> getArcsToFix(const Individual& mpIndividual) const override;

    public:
        explicit RandomPathDecomposition(Params& p) : ArcBasedDecomposition{p} {}
        ~RandomPathDecomposition() override = default;
    };
}

#endif //GENVRP_RANDOMPATHDECOMPOSITION_H
