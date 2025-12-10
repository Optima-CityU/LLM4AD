#include "RandomArcDecomposition.h"

#include <random> // For std::random_shuffle

namespace genvrp {
    std::vector<ArcBasedDecomposition::Arc> RandomArcDecomposition::getArcsToFix(const Individual& mpIndividual) const {
        std::vector<Arc> arcs;

        // Get the list of arcs used in routes, excluding to/from the depot.
        for(const auto& route : mpIndividual.routes) {
            if(route.empty()) {
                continue;
            }

            for(auto i = 0u; i < route.size() - 1u; ++i) {
                arcs.emplace_back(route[i], route[i + 1]);
            }
        }

        // Randomly shuffle the list of arcs.
        random_shuffle(arcs.begin(), arcs.end());

        return arcs;
    }
} // namespace genvrp