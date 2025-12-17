#include "BarycentreQuadrantDecomposition.h"

namespace genvrp {
    RouteSequenceDecomposition::SPList BarycentreQuadrantDecomposition::getSubproblems(const Individual& mpIndividual, const SolverStatus& status) const {
#ifndef BATCH_MODE
        std::cout << status.outPrefix << "[Decomposition] Starting barycentre quadrant decomposition.\n";
#endif

        // Quadrant decomposition only makes sense once (at 1 level or recursion).
        if(status.recursionLevel > 1u) {
#ifndef BATCH_MODE
            std::cout << status.outPrefix << "[Decomposition] Skipping because quadrant decomposition can only be applied once.\n";
#endif
            return {};
        }

        SPList sps;
        std::vector<std::vector<int>> routesByQuadrant(4);

        assert(mpIndividual.routes.size() == params.data.nbVehicles);
        assert(mpIndividual.barycentres.size() == params.data.nbVehicles);

        // Scan the route and put the corresponding index in the correct quadrant.
        for(auto r = 0; r < params.data.nbVehicles; ++r) {
            if(mpIndividual.routes[r].empty()) {
                continue; // Skip empty routes
            }

            const auto& bary = mpIndividual.barycentres[r];

            if(bary.x >= 0) {
                if(bary.y >= 0) {
                    routesByQuadrant[0].push_back(r);
                } else {
                    routesByQuadrant[3].push_back(r);
                }
            } else {
                if(bary.y >= 0) {
                    routesByQuadrant[1].push_back(r);
                } else {
                    routesByQuadrant[2].push_back(r);
                }
            }
        }

        // Create the subproblems.
        for(const auto& quadrantRoutes : routesByQuadrant) {
            if(!quadrantRoutes.empty()) {
                sps.push_back(std::make_unique<SubProblem>(params, mpIndividual, status, quadrantRoutes));
            }
        }

        return sps;
    }
} // namespace genvrp