//
// Created by alberto on 15/03/19.
//

#include "BarycentreSwipeDecomposition.h"

#include <algorithm> // For std::is_sorted, std::rotate
#include <numeric>   // for std::iota

namespace genvrp {

    RouteSequenceDecomposition::SPList BarycentreSwipeDecomposition::getSubproblems(const Individual& mpIndividual, const SolverStatus& status) const {
#ifndef BATCH_MODE
        std::cout << status.outPrefix << "[Decomposition] Starting barycentre swipe decomposition.\n";
#endif

        auto mpRoutesIndices = std::vector<int>(mpIndividual.routes.size());
        std::iota(mpRoutesIndices.begin(), mpRoutesIndices.end(), 0);

        // The routes in the individual should already be sorted by barycentre angle.
        assert(std::is_sorted(mpRoutesIndices.begin(), mpRoutesIndices.end(),
                              [&](const int& i1, const int& i2) -> bool { return mpIndividual.barycentres[i1].angle < mpIndividual.barycentres[i2].angle; }));

        // Then just select a random starting point and rotate.
        const auto start = std::rand() % mpRoutesIndices.size();
        std::rotate(mpRoutesIndices.begin(), mpRoutesIndices.begin() + start, mpRoutesIndices.end());

        return getSubproblemsFromRouteIndices(mpIndividual, status, mpRoutesIndices);
    }
} // namespace genvrp