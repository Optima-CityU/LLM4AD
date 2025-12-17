//
// Created by alberto on 14/03/19.
//

#include "RandomRouteDecomposition.h"

#include <iostream> // For std::cout
#include <numeric>  // For std::iota

namespace genvrp {
    RouteSequenceDecomposition::SPList RandomRouteDecomposition::getSubproblems(const Individual& mpIndividual, const SolverStatus& status) const {
#ifndef BATCH_MODE
        std::cout << status.outPrefix << "[Decomposition] Starting random route decomposition.\n";
#endif

        auto mpRoutesIndices = std::vector<int>(mpIndividual.routes.size());
        std::iota(mpRoutesIndices.begin(), mpRoutesIndices.end(), 0);
        random_shuffle(mpRoutesIndices.begin(), mpRoutesIndices.end());

        return getSubproblemsFromRouteIndices(mpIndividual, status, mpRoutesIndices);
    }
} // namespace genvrp