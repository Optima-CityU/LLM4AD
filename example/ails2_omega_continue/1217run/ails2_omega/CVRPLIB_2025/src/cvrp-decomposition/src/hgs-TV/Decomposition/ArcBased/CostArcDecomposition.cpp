#include "CostArcDecomposition.h"

#include <algorithm> // For std::sort
#include <vector>

namespace genvrp {
    std::vector<ArcBasedDecomposition::Arc> CostArcDecomposition::getArcsToFix(const Individual& mpIndividual) const {
        std::vector<Arc> arcs;

        // Get the list of arcs used in routes, excluding to/from the depot.
        for(const auto& route : mpIndividual.routes) {
            if(route.empty()) {
                continue;
            }

            for(auto i = 0u; i < route.size() - 1u; ++i) {
                const auto orig = route[i];
                const auto dest = route[i + 1];

                arcs.emplace_back(orig, dest, params.data.timeCost[orig][dest]);
            }
        }

        // Get the shortest arcs.
        std::sort(arcs.begin(), arcs.end(), [](const Arc& a1, const Arc& a2) { return a1.cost < a2.cost; });

        return arcs;
    }
} // namespace genvrp