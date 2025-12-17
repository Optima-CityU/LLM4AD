#include "RandomPathDecomposition.h"

#include <random> // For std::random_shuffle

namespace genvrp {
    std::vector<ArcBasedDecomposition::Arc> RandomPathDecomposition::getArcsToFix(const Individual& mpIndividual) const {
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

        // Randomly shuffle the list of paths.
        random_shuffle(paths.begin(), paths.end());

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