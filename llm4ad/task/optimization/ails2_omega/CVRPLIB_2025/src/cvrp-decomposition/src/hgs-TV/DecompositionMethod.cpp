//
// Created by alberto on 12/03/19.
//

#include "DecompositionMethod.h"
#include "Decomposition/ArcBased/ArcHistoryDecomposition.h"
#include "Decomposition/ArcBased/CostArcDecomposition.h"
#include "Decomposition/ArcBased/CostPathDecomposition.h"
#include "Decomposition/ArcBased/PathHistoryDecomposition.h"
#include "Decomposition/ArcBased/RandomArcDecomposition.h"
#include "Decomposition/ArcBased/RandomPathDecomposition.h"
#include "Decomposition/RouteSequence/BarycentreClusteringDecomposition.h"
#include "Decomposition/RouteSequence/BarycentreQuadrantDecomposition.h"
#include "Decomposition/RouteSequence/BarycentreSwipeDecomposition.h"
#include "Decomposition/RouteSequence/RandomRouteDecomposition.h"
#include "Decomposition/RouteSequence/RouteHistoryDecomposition.h"

namespace genvrp {
    std::unique_ptr<DecompositionMethod> getDecompositionMethod(Params& params) {
        switch(params.deco.type) {
        case Params::DecompositionType::None: return nullptr;
        case Params::DecompositionType::RandomRoute: return std::make_unique<RandomRouteDecomposition>(params);
        case Params::DecompositionType::BarycentreSwipe: return std::make_unique<BarycentreSwipeDecomposition>(params);
        case Params::DecompositionType::BarycentreQuadrant: return std::make_unique<BarycentreQuadrantDecomposition>(params);
        case Params::DecompositionType::BarycentreClustering: return std::make_unique<BarycentreClusteringDecomposition>(params);
        case Params::DecompositionType::RouteHistory: return std::make_unique<RouteHistoryDecomposition>(params);
        case Params::DecompositionType::RandomArc: return std::make_unique<RandomArcDecomposition>(params);
        case Params::DecompositionType::CostArc: return std::make_unique<CostArcDecomposition>(params);
        case Params::DecompositionType::ArcHistory: return std::make_unique<ArcHistoryDecomposition>(params);
        case Params::DecompositionType::RandomPath: return std::make_unique<RandomPathDecomposition>(params);
        case Params::DecompositionType::CostPath: return std::make_unique<CostPathDecomposition>(params);
        case Params::DecompositionType::PathHistory: return std::make_unique<PathHistoryDecomposition>(params);

        default:
            // We forgot to implement the case for some enum value!
            assert(false);
            return nullptr;
        }
    }

    void DecompositionMethod::mergeSpSolutionIntoMpIndividual(const Individual& spSolution, Individual& mpIndividual, const SubProblem& subproblem) const {
        // Check if the subproblem solution improved on the master problem one.
        double eliteCost = 0.0;

        for(const auto& route_id : subproblem.routes) {
            assert(eliteIndividual);
            const auto& route = eliteIndividual->routes[route_id];

            if(route.empty()) {
                continue;
            }

            eliteCost += params.data.timeCost[0][route[0u]];

            for(auto cust_pos = 0u; cust_pos < route.size() - 1u; ++cust_pos) {
                eliteCost += params.data.timeCost[route[cust_pos]][route[cust_pos + 1u]];
            }

            eliteCost += params.data.timeCost[route.back()][0];
        }

        if(eliteCost > spSolution.cost.penalizedCost) {
#ifndef BATCH_MODE
            std::cerr << subproblem.spStatus.outPrefix << "[Decomposition] SP solution better than MP's. Using SP solution.\n";
#endif

            // Use the individual from the subproblem.
            for(const auto& route : spSolution.routes) {
                // Add a new route to the MP individual.
                mpIndividual.routes.emplace_back();

                for(const auto& spCustomer : route) {
                    // Get the MP customer mapping to the given SP customer.
                    const auto mpCustomer = subproblem.spToMpCustomers.at(spCustomer);

                    // Add the MP customer to routes and giantTour.
                    mpIndividual.routes.back().push_back(mpCustomer);
                    mpIndividual.giantTour.push_back(mpCustomer);
                }
            }

            // New route strictly better than old part.
            params.stats.nRouteSpImproved += 1u;
        } else {
#ifndef BATCH_MODE
            std::cerr << subproblem.spStatus.outPrefix << "[Decomposition] SP solution worse than MP's. Keeping MP solution.\n";
#endif

            // Use the elite individual from the master problem.
            for(const auto& route_id : subproblem.routes) {
                mpIndividual.routes.emplace_back();

                for(const auto& mpCustomer : eliteIndividual->routes[route_id]) {
                    mpIndividual.routes.back().push_back(mpCustomer);
                    mpIndividual.giantTour.push_back(mpCustomer);
                }
            }

            // New route worse or equal than starting part.
            params.stats.nRouteSpWorsened += 1u;
        }
    }
} // namespace genvrp