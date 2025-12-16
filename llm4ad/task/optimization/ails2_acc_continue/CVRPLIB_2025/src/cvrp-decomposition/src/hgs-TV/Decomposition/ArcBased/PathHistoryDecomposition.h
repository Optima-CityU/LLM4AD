#ifndef GENVRP_PATHHISTORYDECOMPOSITION_H
#define GENVRP_PATHHISTORYDECOMPOSITION_H

#include "../ArcBasedDecomposition.h"

namespace genvrp {
    class PathHistoryDecomposition : public ArcBasedDecomposition {
    protected:
        /** Given a MP individual, gets the 30% of arcs which have been used the most historically. */
        std::vector<Arc> getArcsToFix(const Individual& mpIndividual) const override;

    public:
        explicit PathHistoryDecomposition (Params& p) : ArcBasedDecomposition{p} {}
        ~PathHistoryDecomposition () override = default;
    };
}


#endif //GENVRP_PATHHISTORYDECOMPOSITION_H