#ifndef GENVRP_ARCHISTORYDECOMPOSITION_H
#define GENVRP_ARCHISTORYDECOMPOSITION_H

#include "../ArcBasedDecomposition.h"

namespace genvrp {
    class ArcHistoryDecomposition : public ArcBasedDecomposition {
    protected:
        /** Given a MP individual, gets the 40% of arcs which have been used the most historically. */
        std::vector<Arc> getArcsToFix(const Individual& mpIndividual) const override;

    public:
        explicit ArcHistoryDecomposition(Params& p) : ArcBasedDecomposition{p} {}
        ~ArcHistoryDecomposition() override = default;
    };
}

#endif //GENVRP_ARCHISTORYDECOMPOSITION_H
