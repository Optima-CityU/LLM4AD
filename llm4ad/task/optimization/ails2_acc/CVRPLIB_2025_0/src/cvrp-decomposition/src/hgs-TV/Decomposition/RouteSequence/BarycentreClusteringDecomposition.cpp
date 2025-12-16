#include "BarycentreClusteringDecomposition.h"
#include "../ClusteringUtils/KMeans.h"

namespace genvrp {

    RouteSequenceDecomposition::SPList BarycentreClusteringDecomposition::getSubproblems(const Individual& mpIndividual, const SolverStatus& status) const {
        using namespace kmeans;

#ifndef BATCH_MODE
        std::cout << status.outPrefix << "[Decomposition] Starting barycentre clustering decomposition.\n";
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

        // Get the clustering of the routes' barycentres via k-means.
        const auto clustering = kMeans(k, mpIndividual.barycentres, emptyRoutes);
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
} // namespace genvrp