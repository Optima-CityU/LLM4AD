#include "ArcHistoryDecomposition.h"

namespace genvrp {
    std::vector<ArcBasedDecomposition::Arc> ArcHistoryDecomposition::getArcsToFix(const Individual& mpIndividual) const {
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

        assert(params.afreq);

        // Get the arcs with highest values in afreq.
        std::sort(arcs.begin(), arcs.end(),
                  [this](const Arc& a1, const Arc& a2) { return (*params.afreq)[a1.origin][a1.destination] > (*params.afreq)[a2.origin][a2.destination]; });

        return arcs;
    }
} // namespace genvrp