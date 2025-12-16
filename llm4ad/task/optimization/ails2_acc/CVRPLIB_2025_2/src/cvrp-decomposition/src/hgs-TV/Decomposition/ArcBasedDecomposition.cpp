//
// Created by alberto on 20/03/19.
//

#include <algorithm> // For std::remove_if
#include <cassert>
#include <memory> // For std::make_unique
#include <set>
#include <type_traits> // For std::is_same_t
#include <utility>     // For std::make_pair

#include "../Genetic.h"
#include "../Individual.h"
#include "../Params.h"
#include "ArcBasedDecomposition.h"

namespace genvrp {

    namespace {
        bool isSubPath(const ArcBasedDecomposition::Path& path, const ArcBasedDecomposition::Arc& arc) {
            if(path.size() < 2ul) {
                return false;
            }

            for(auto pos = 0u; pos < path.size() - 1u; ++pos) {
                // Arc is already part of this path.
                if(arc.origin == path[pos] && arc.destination == path[pos + 1u]) {
                    return true;
                }

                // The reverse arc is already part of this path.
                if(arc.destination == path[pos] && arc.origin == path[pos + 1u]) {
                    return true;
                }
            }

            return false;
        }

        bool isSubPathOfAny(const std::vector<ArcBasedDecomposition::Path>& paths, const ArcBasedDecomposition::Arc& arc) {
            return std::any_of(paths.begin(), paths.end(), [&arc](const auto& path) -> bool { return isSubPath(path, arc); });
        }

        bool attachesToPath(ArcBasedDecomposition::Path& path, const ArcBasedDecomposition::Arc& arc, ArcBasedDecomposition::CustomerSet& custs) {
            if(arc.destination == path.front()) {
                // I have a path v1, v2, v3, ...
                // and now I consider an arc (v0, v1)
                // The path needs to become v0, v1, v2, ...
                path.insert(path.begin(), arc.origin);
                custs.push_back(arc.origin);
                return true;
            } else if(arc.origin == path.back()) {
                // I have a path ..., v98, v99, v100
                // and now I consider an arc (v100, 101)
                // The path needs to become ..., v99, v100, v101
                path.push_back(arc.destination);
                custs.push_back(arc.destination);
                return true;
            }

            return false;
        }

        bool attachesToAnyPath(std::vector<ArcBasedDecomposition::Path>& paths, const ArcBasedDecomposition::Arc& arc,
                               ArcBasedDecomposition::CustomerSet& custs) {
            return std::any_of(paths.begin(), paths.end(), [&arc, &custs](auto& path) -> bool { return attachesToPath(path, arc, custs); });
        }

        void mergePaths(std::vector<ArcBasedDecomposition::Path>& paths, ArcBasedDecomposition::CustomerSet& custs) {
            // Check if there are any paths to merge.
            // For example, if before I had two paths
            // P1: v0, v1, v2
            // P2: v3, v4, v5
            // And the new arc is (v2, v3), I would now have, e.g.:
            // P1: v0, v1, v2, v3
            // P2: v3, v4, v5
            // This needs to be merged into:
            // P1: v0, v1, v2, v3, v4, v5

            for(auto p1 = 0ul; p1 < paths.size(); ++p1) {
                if(paths[p1].empty()) {
                    continue;
                }
                bool merged = false;

                for(auto p2 = p1 + 1ul; p2 < paths.size(); ++p2) {
                    if(paths[p2].empty()) {
                        continue;
                    }

                    if(paths[p1].front() == paths[p2].back()) {
                        const auto link = paths[p1].front();

                        paths[p2].insert(paths[p2].end(), paths[p1].begin() + 1, paths[p1].end());
                        paths[p1].clear();

                        // The "linking" customer was counted twice.
                        if(std::count(custs.begin(), custs.end(), link) > 1u) {
                            custs.erase(std::find(custs.begin(), custs.end(), link));
                        }

                        merged = true;
                        break;
                    }

                    if(paths[p1].back() == paths[p2].front()) {
                        const auto link = paths[p1].back();

                        paths[p1].insert(paths[p1].end(), paths[p2].begin() + 1, paths[p2].end());
                        paths[p2].clear();

                        // The "linking" customer was counted twice.
                        if(std::count(custs.begin(), custs.end(), link) > 1u) {
                            custs.erase(std::find(custs.begin(), custs.end(), link));
                        }

                        merged = true;
                        break;
                    }
                }

                if(merged) {
                    break;
                }
            }
        }
    } // namespace

    std::pair<ArcBasedDecomposition::CustomerSet, std::vector<ArcBasedDecomposition::Path>>
    ArcBasedDecomposition::getPaths(const std::vector<ArcBasedDecomposition::Arc>& arcs) const {
        CustomerSet custs;
        std::vector<Path> paths;

        for(const auto& arc : arcs) {
            // If the arc is already in some path, skip it.
            if(isSubPathOfAny(paths, arc)) {
                continue;
            }

            if(attachesToAnyPath(paths, arc, custs)) {
                // Check if there are any paths to merge.
                mergePaths(paths, custs);
            } else {
                // Start a new path.
                paths.push_back({arc.origin, arc.destination});
                custs.push_back(arc.origin);
                custs.push_back(arc.destination);
            }

            // Check there are no duplicates.
            assert(std::set(custs.begin(), custs.end()).size() == custs.size());

            // Clean up empty paths.
            paths.erase(std::remove_if(paths.begin(), paths.end(), [](const auto& p) { return p.empty(); }), paths.end());
        }

        return std::make_pair(custs, paths);
    }

    void ArcBasedDecomposition::operator()(Population& population, const SolverStatus& status) {
        // If the problem does not have enough individuals, skip decomposition.
        if(params.data.nbClients <= params.deco.targetMaxSpCustomers) {
#ifndef BATCH_MODE
            std::cout << status.outPrefix << "[Decomposition] Not enough customers for decomposition (" << params.data.nbClients << " < "
                      << params.deco.targetMaxSpCustomers << ")\n";
#endif

            return;
        }

        // If decomposition is too nested, skip decomposition.
        if(status.recursionLevel > params.deco.maxDecoLevel) {
#ifndef BATCH_MODE
            std::cout << status.outPrefix << "[Decomposition] Too nested (" << status.recursionLevel << " > " << params.deco.maxDecoLevel << ")\n";
#endif

            return;
        }

        // Get an elite individual.
        eliteIndividual.emplace(population.getEliteIndividual(params.deco.eliteFraction));

        // Get the arcs to fix.
        auto arcs = getArcsToFix(*eliteIndividual);

        // Remove eventual duplicate arcs.
        std::sort(arcs.begin(), arcs.end());
        arcs.erase(std::unique(arcs.begin(), arcs.end()), arcs.end());

        // Check there are no duplicate arcs.
        assert(std::set(arcs.begin(), arcs.end()).size() == arcs.size());

        CustomerSet custsInPaths;
        std::vector<Path> paths;
        std::size_t nArcsUsed = 1u;

        // Keep accumulating from arcs until we reach the correct subproblem size.
        // In the subproblem there will be one vertex for each path, plus all other vertices
        // corresponding to original non-shrunk customers.
        while(paths.size() + (params.data.nbClients - custsInPaths.size()) > params.deco.targetMaxSpCustomers && nArcsUsed < arcs.size()) {
            const auto arcsUsed = std::vector<Arc>(arcs.begin(), arcs.begin() + nArcsUsed);

            // Check there are no duplicate arcs.
            assert(std::set(arcsUsed.begin(), arcsUsed.end()).size() == arcsUsed.size());

            std::tie(custsInPaths, paths) = getPaths(arcsUsed);
            ++nArcsUsed;

            // Check all elements are unique.
            assert(std::set(custsInPaths.begin(), custsInPaths.end()).size() == custsInPaths.size());
        }

        // Customers not visited by the arcs.
        CustomerSet custsNotInPaths;
        custsNotInPaths.reserve(params.data.nbClients - custsInPaths.size());
        for(auto i = 1; i <= params.data.nbClients; ++i) {
            if(std::find(custsInPaths.begin(), custsInPaths.end(), i) == custsInPaths.end()) {
                custsNotInPaths.push_back(i);
            }
        }

        assert(custsInPaths.size() + custsNotInPaths.size() == params.data.nbClients);

        // Create a mapping between customers in the new subproblem and the corresponding sets of customers in the
        // master problem. Also create the reverse map.
        std::map<int, CustomerSet> spToMp;
        std::map<int, int> mpToSp;

        // The depot stays the same
        spToMp[0] = {0};
        mpToSp[0] = 0;

        int custNumber = 1;
        int addedCustomers = 0;

        // First add the singletons (customers not in paths).
        for(const auto& c : custsNotInPaths) {
            spToMp[custNumber] = {c};
            mpToSp[c] = custNumber;
            ++custNumber;
            ++addedCustomers;
        }

        // Then add the customers in paths.
        static_assert(std::is_same_v<CustomerSet, Path>);
        for(const auto& path : paths) {
            assert(!path.empty());

            spToMp[custNumber] = path;

            for(const auto& mpCust : path) {
                mpToSp[mpCust] = custNumber;
            }

            ++custNumber;
            addedCustomers += path.size();
        }

        // Make sure all master problem customers are placed somewhere.
        assert(addedCustomers == params.data.nbClients);

        // Create the params object for the subproblem.
        // At first, copy-construct from the master problem params.
        Params spParams{params};

        // Number of clients in the subproblem. Since there is one subproblem
        // client for each key in spToMp, we can get this number from there.
        // We subtract 1 to account for the depot, which should not be counted.
        spParams.data.nbClients = spToMp.size() - 1;

        /*
         *  There are 2 ways of assigning travel and service times in the subproblem:
         *      1. Service time of a SP node ~ path = sum of the service times plus the
         *      travel times in the path; travel time to a SP node ~ path = travel time
         *      to the first MP node of the path; travel time from a SP node ~ path =
         *      travel time from the last MP node of the path.
         *
         *      2. Service time of a SP node ~ path = sum of the service times in the
         *      path; travel time to a SP node ~ path = travel time to the first MP node
         *      of the path + 1/2 of the travel time of the path; travel time from a SP
         *      node ~ path = travel time from the last MP node of the path + 1/2 of the
         *      travel time of the path.
         *
         *      Of this two method we prefer the 1st one, to avoid messing with granular
         *      sparsification of the graph in the SP.
         */

        // Create the vector of clients for the subproblem.
        spParams.data.cli = std::vector<Params::Client>(spParams.data.nbClients + 1);
        spParams.data.cli[0] = params.data.cli[0];
        // At the same time, compute the maximum demand.
        spParams.data.maxDemand = 0.0;

        // You don't need the following vector if implementing point 1. above.
        // std::vector<double> pathTravelTimes(spToMp.size(), 0.0);

        for(const auto& [c, path] : spToMp) {
            Params::Client client;

            client.custNum = c;
            client.coordX = 0; // Not needed
            client.coordY = 0; // Not needed

            // As a service duration we put the actual service durations of the
            // customers in path, plus the travel time within path customers.
            // As demand we put the sum of the demands.
            client.serviceDuration = 0.0;
            client.demand = 0.0;
            for(auto i = 0u; i < path.size(); ++i) {
                const auto mpCust = path[i];
                client.serviceDuration += params.data.cli[mpCust].serviceDuration;

                if(i < path.size() - 1u) {
                    const auto nextMpCust = path[i + 1];
                    // This line implements point 1. above.
                    client.serviceDuration += params.data.timeCost[mpCust][nextMpCust];
                    // This line implements point 2. above.
                    // pathTravelTimes[c] += params.data.timeCost[mpCust][nextMpCust];
                }

                client.demand += params.data.cli[mpCust].demand;
            }

            spParams.data.maxDemand = std::max(spParams.data.maxDemand, client.demand);
            spParams.data.cli[c] = client;
        }

        // Create the matrix of travel times.
        // Travel time to a path is travel time to the beginning of the path.
        // Travel time from a path is travel time from the end of the path.
        spParams.data.timeCost = std::vector<std::vector<double>>(spParams.data.nbClients + 1, std::vector<double>(spParams.data.nbClients + 1, 0));

        // At the same time, compute the maximum distance.
        spParams.data.maxDist = 0.0;

        for(auto c1 = 0; c1 <= spParams.data.nbClients; ++c1) {
            // Origin vertex => take the last element of the path.
            // If the subproblem customer is not a path, it still works
            // because spToMp[c1] will be a singleton. This also works
            // for the depot, which is singleton {0}.
            const auto& mp_c1 = spToMp[c1].back();

            for(auto c2 = 0; c2 <= spParams.data.nbClients; ++c2) {
                if(c1 == c2) {
                    continue;
                }

                // Destination vertex => take the first element of the path.
                // If the subproblem customer is not a path, it still works
                // because spToMp[c1] will be a singleton. This also works
                // for the depot, which is singleton {0}.
                const auto& mp_c2 = spToMp[c2].front();

                auto dist = params.data.timeCost[mp_c1][mp_c2];

                // These lines implement solution 2. above.
                // dist += pathTravelTimes[c1] / 2.0;
                // dist += pathTravelTimes[c2] / 2.0;

                spParams.data.timeCost[c1][c2] = dist;

                spParams.data.maxDist = std::max(spParams.data.maxDist, dist);
            }
        }

        // Reset the frequency matrices.
        if(spParams.vfreq) {
            spParams.vfreq = Params::FrequencyMatrix(spParams.data.nbClients + 1, std::vector<std::size_t>(params.data.nbClients + 1, 0u));
        }
        if(spParams.afreq) {
            spParams.afreq = Params::FrequencyMatrix(params.data.nbClients + 1, std::vector<std::size_t>(params.data.nbClients + 1, 0u));
        }

        // Recompute correlated vertices.
        spParams.setupDataCorrelatedVertices();

        // Set the subproblem termination criteria.
        spParams.ga.maxIter = params.deco.spMaxIter;
        spParams.ga.maxIterNonProd = params.deco.spMaxIterNonProd;

        // Set the subproblem population criteria.
        spParams.ga.nbElite = params.deco.spNbElite;
        spParams.ga.lambda = params.deco.spLambda;
        spParams.ga.mu = params.deco.spMu;

        // Create the initial individual.
        Individual spIndividual{&spParams};
        spIndividual.routes.clear();
        spIndividual.giantTour.clear();

        for(const auto& route : eliteIndividual->routes) {
            spIndividual.routes.emplace_back();

            for(const auto& mpCust : route) {
                auto& currentSpRoute = spIndividual.routes.back();
                const auto& spCust = mpToSp[mpCust];

                // Avoid repeating the same sp customer more than once when
                // following a path which was shrunk in the subproblem.
                if(currentSpRoute.empty() || currentSpRoute.back() != spCust) {
                    currentSpRoute.push_back(spCust);
                    spIndividual.giantTour.push_back(spCust);
                }
            }
        }

        spIndividual.evaluateCompleteCost();

        // Check that indeed the new sp individual has the same cost than the mp elite individual used to generate it.
        assert(std::abs(eliteIndividual->cost.penalizedCost - spIndividual.cost.penalizedCost) < 1e-3);

#ifndef BATCH_MODE
        std::cout << status.outPrefix << "[Decomposition] Subproblem has " << spParams.data.nbClients << " customers.\n";
#endif

        // Solve the subproblem.
        auto subproblem = Genetic{spParams};
        const auto spSolution = subproblem.run(status);

        // Add the subproblem stats to the master problem's.
        params.stats.spTime += spParams.stats.spTime;
        params.stats.nSpIterations += spParams.stats.nSpIterations;

        if(!spSolution) {
#ifndef BATCH_MODE
            std::cout << status.outPrefix << "[Decomposition] No solution found in the subproblem.\n";
#endif

            return;
        }

        // New MP individual we want to add.
        auto mpIndividual = Individual{&params};
        mpIndividual.routes.clear();
        mpIndividual.giantTour.clear();

        for(const auto& spRoute : spSolution->routes) {
            mpIndividual.routes.emplace_back();

            for(const auto& spCust : spRoute) {
                for(const auto& mpCust : spToMp[spCust]) {
                    mpIndividual.routes.back().push_back(mpCust);
                    mpIndividual.giantTour.push_back(mpCust);
                }
            }
        }

        assert(mpIndividual.routes.size() == params.data.nbVehicles);
        assert(mpIndividual.giantTour.size() == params.data.nbClients);

        mpIndividual.evaluateCompleteCost();

#ifndef BATCH_MODE
        std::cout << status.outPrefix << "[Decomposition] Obtained new solution of cost " << mpIndividual.cost.penalizedCost << "\n";
#endif

        if(mpIndividual.cost.penalizedCost >= eliteIndividual->cost.penalizedCost) {
            // New solution worse, or same, than elite starting one.
            params.stats.nSpWorsened += 1u;
        } else if(mpIndividual.cost.penalizedCost < eliteIndividual->cost.penalizedCost) {
            // New solution strictly better than elite starting one.
            params.stats.nSpImproved += 1u;
        }

        // Add the new individual to the MP population.
        population.addIndividualLS(&mpIndividual, true);
    }
} // namespace genvrp