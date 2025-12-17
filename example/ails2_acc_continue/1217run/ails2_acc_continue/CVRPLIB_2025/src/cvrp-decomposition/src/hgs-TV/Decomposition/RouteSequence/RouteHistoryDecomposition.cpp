#include "RouteHistoryDecomposition.h"
#include "../ClusteringUtils/KMedoids.h"

#include <cmath> // For std::ceil
#include <limits>
#include <memory> // For std::make_unique
#include <vector>

namespace genvrp {
    RouteSequenceDecomposition::SPList RouteHistoryDecomposition::getSubproblems(const Individual& mpIndividual, const SolverStatus& status) const {
        using namespace kmedoids;

#ifndef BATCH_MODE
        std::cout << status.outPrefix << "[Decomposition] Starting route history decomposition.\n";
#endif

        // Number of clusters to create (i.e., number of subproblems to create).
        const auto k = (std::size_t)std::ceil((double)params.data.nbClients / (double)params.deco.targetMaxSpCustomers);

        // Determine which routes are empty, as they should not be considered during clustering
        // (what's their barycentre?).
        auto emptyRoutes = std::vector<int>{};
        for(auto i = 0u; i < mpIndividual.routes.size(); ++i) {
            if(mpIndividual.routes[i].empty()) {
                emptyRoutes.push_back(i);
            }
        }

        // Get the routes scores, computed as score(i, j) =
        // sum(custi in route[i]) sum(custj in route[j]) {
        //   params.vfreq[i][j]
        // }
        const auto routeDistances = getDistances(mpIndividual);

        // Get the clustering of the routes' barycentres via k-medoids.
        const auto clustering = kMedoids(k, routeDistances, emptyRoutes);
        assert(clustering.size() == k);

        // We will now proceed to create k subproblems, one for each cluster.
        auto sps = std::vector<std::unique_ptr<SubProblem>>{};
        sps.reserve(k);

        // Cycle through each cluster. A cluster is a collection of indices
        // which index routes in base_individual.
        for(const auto& cluster : clustering) {
            if(!cluster.empty()) {
                // Create a subproblem with the routes indexed by the cluster.
                sps.push_back(std::make_unique<SubProblem>(params, mpIndividual, status, cluster));
            }
        }

        return sps;
    }

    double RouteHistoryDecomposition::getRoutesDistance(const std::vector<int>& route1, const std::vector<int>& route2) const {
        assert(params.vfreq);

        double score = 0.0f;

        for(const auto& c1 : route1) {
            for(const auto& c2 : route2) {
                score += (*params.vfreq)[c1][c2];
            }
        }

        if(score == 0.0) {
            return std::numeric_limits<double>::max();
        } else {
            return 1 / score;
        }
    }

    std::vector<std::vector<double>> RouteHistoryDecomposition::getDistances(const Individual& mpIndividual) const {
        const auto& r = mpIndividual.routes;
        std::vector<std::vector<double>> distances(r.size(), std::vector<double>(r.size()));

        for(auto i = 0u; i < r.size(); ++i) {
            for(auto j = 0u; j < r.size(); ++j) {
                distances[i][j] = getRoutesDistance(r[i], r[j]);
                distances[j][i] = distances[i][j];
            }
        }

        return distances;
    }
} // namespace genvrp