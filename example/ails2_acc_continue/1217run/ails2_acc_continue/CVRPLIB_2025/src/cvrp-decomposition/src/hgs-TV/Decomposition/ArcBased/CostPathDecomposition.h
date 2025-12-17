#ifndef GENVRP_COSTPATHDECOMPOSITION_H
#define GENVRP_COSTPATHDECOMPOSITION_H

#include "../ArcBasedDecomposition.h"
#include "../../Individual.h"
#include "../../Params.h"

#include <vector>

namespace genvrp {
    class CostPathDecomposition : public ArcBasedDecomposition {
    protected:
        /** Given a MP individual, gets the shortest 40% of its arcs. */
        std::vector<Arc> getArcsToFix(const Individual& mpIndividual) const override;

    public:
        explicit CostPathDecomposition(Params& p) : ArcBasedDecomposition{p} {}
        ~CostPathDecomposition() override = default;
    };
}

#endif //GENVRP_COSTPATHDECOMPOSITION_H
