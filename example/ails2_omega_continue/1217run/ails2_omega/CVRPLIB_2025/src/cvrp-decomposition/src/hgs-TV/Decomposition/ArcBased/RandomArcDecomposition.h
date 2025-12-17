#ifndef GENVRP_RANDOMARCDECOMPOSITION_H
#define GENVRP_RANDOMARCDECOMPOSITION_H

#include "../ArcBasedDecomposition.h"
#include "../../Params.h"

#include <vector>

namespace genvrp {
    class RandomArcDecomposition : public ArcBasedDecomposition {
    protected:
        /** Given a MP individual, gets 40% of its arcs at random. */
        std::vector<Arc> getArcsToFix(const Individual& mpIndividual) const override;

    public:
        explicit RandomArcDecomposition(Params& p) : ArcBasedDecomposition{p} {}
        ~RandomArcDecomposition() override = default;
    };
}

#endif //GENVRP_RANDOMARCDECOMPOSITION_H
