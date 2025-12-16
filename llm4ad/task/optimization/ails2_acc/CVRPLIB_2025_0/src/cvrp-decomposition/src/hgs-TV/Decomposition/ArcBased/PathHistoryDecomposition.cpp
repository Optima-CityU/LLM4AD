//
// Created by Alberto Santini on 11/11/2021.
//

#include "PathHistoryDecomposition.h"

namespace genvrp {
    std::vector<ArcBasedDecomposition::Arc> PathHistoryDecomposition::getArcsToFix(const Individual& mpIndividual) const {
        std::vector<Path> paths;

        // Get the list of paths used in routes, excluding to/from the depot.
        for(const auto& route : mpIndividual.routes) {
            if(route.empty()) {
                continue;
            }

            for(auto it = route.begin(); it <= route.end() - Params::record_path_size; ++it) {
                const auto path = std::vector<int>(it, it + Params::record_path_size);
                paths.push_back(std::move(path));
            }
        }

        assert(params.pfreq);

        std::sort(paths.begin(), paths.end(), [&](const Path& p1, const Path& p2) -> bool {
            auto it1 = params.pfreq->find(p1);
            if(it1 == params.pfreq->end()) {
                return false;
            }

            auto it2 = params.pfreq->find(p2);
            if(it2 == params.pfreq->end()) {
                return true;
            }

            return it1->second < it2->second;
        });

        // Decompose the paths into single arcs.
        std::vector<Arc> arcs;

        while(!paths.empty()) {
            const auto path = paths.back();
            paths.pop_back();

            for(auto i = 0u; i < path.size(); ++i) {
                if(i < path.size() - 1u) {
                    arcs.emplace_back(path[i], path[i + 1]);
                }

                paths.erase(std::remove(paths.begin(), paths.end(), path), paths.end());
            }
        }

        // Needed because in the while loop above we go through paths from its back()
        std::reverse(arcs.begin(), arcs.end());

        return arcs;
    }
} // namespace genvrp