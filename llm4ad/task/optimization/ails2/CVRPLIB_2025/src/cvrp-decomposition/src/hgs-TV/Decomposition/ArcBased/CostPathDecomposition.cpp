#include "CostPathDecomposition.h"

#include <utility>

namespace genvrp {
    std::vector<ArcBasedDecomposition::Arc> CostPathDecomposition::getArcsToFix(const Individual& mpIndividual) const {
        using CostPath = std::pair<double, Path>;
        std::vector<CostPath> cpaths;

        // Get the list of paths used in routes, excluding to/from the depot.
        for(const auto& route : mpIndividual.routes) {
            if(route.empty()) {
                continue;
            }

            for(auto it = route.begin(); it <= route.end() - Params::record_path_size; ++it) {
                const auto path = std::vector<int>(it, it + Params::record_path_size);
                auto cost = 0.0;

                for(auto jt = it; jt < it + Params::record_path_size - 1; ++jt) {
                    cost += params.data.timeCost[*jt][*(jt + 1)];
                }

                cpaths.emplace_back(cost, std::move(path));
            }
        }

        // Sort paths by cost.
        std::sort(cpaths.begin(), cpaths.end());

        // Decompose the paths into single arcs.
        std::vector<Arc> arcs;

        while(!cpaths.empty()) {
            const auto path = cpaths.back().second;
            cpaths.pop_back();

            for(auto i = 0u; i < path.size(); ++i) {
                if(i < path.size() - 1u) {
                    arcs.emplace_back(path[i], path[i + 1]);
                }

                cpaths.erase(std::remove_if(cpaths.begin(), cpaths.end(),
                                            [&](const auto& cp) -> bool { return std::find(cp.second.begin(), cp.second.end(), path[i]) != cp.second.end(); }),
                             cpaths.end());
            }
        }

        // Needed because in the while loop above we go through cpaths from its back()
        std::reverse(arcs.begin(), arcs.end());

        return arcs;
    }
} // namespace genvrp