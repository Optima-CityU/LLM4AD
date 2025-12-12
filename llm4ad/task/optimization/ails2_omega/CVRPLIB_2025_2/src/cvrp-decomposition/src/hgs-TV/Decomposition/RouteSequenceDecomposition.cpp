//
// Created by alberto on 13/03/19.
//

#include "RouteSequenceDecomposition.h"

namespace genvrp {
    void RouteSequenceDecomposition::operator()(genvrp::Population& population, const SolverStatus& status) {
        // If the problem does not have enough individuals, skip decomposition.
        if(params.data.nbClients <= params.deco.targetMaxSpCustomers) {
#ifndef BATCH_MODE
            std::cout << status.outPrefix << "[Decomposition] Not enough customers for decomposition (" << params.data.nbClients << " < "
                      << params.deco.targetMaxSpCustomers << ")\n";
#endif

            return;
        }

        // Get an elite individual.
        eliteIndividual.emplace(population.getEliteIndividual(params.deco.eliteFraction));

        // Get the subproblems to solve. Method getSubproblems needs to be implemented
        // by any class deriving from RouteSequenceDecomposition.
        auto subproblems = getSubproblems(*eliteIndividual, status);

#ifndef BATCH_MODE
        std::cout << status.outPrefix << "[Decomposition] Created " << subproblems.size() << " subproblems.\n";
#endif

        // Skip if there is only one subproblem.
        if(subproblems.size() < 2u) {
#ifndef BATCH_MODE
            std::cout << status.outPrefix << "[Decomposition] Not enough subproblems (<2).\n";
#endif

            return;
        }

        // New individual that we want to build starting from the subproblems solutions.
        auto mpIndividual = Individual{&params};
        mpIndividual.routes.clear();
        mpIndividual.giantTour.clear();

        for(auto& subproblem : subproblems) {
            const std::optional<Individual> spSolution = subproblem->solve();

            if(spSolution) {
                // Add the partial solution obtained in the subproblem to the complete
                // solution in the master problem individual.
                mergeSpSolutionIntoMpIndividual(*spSolution, mpIndividual, *subproblem);
            }
        }

        // Add empty routes in case the number of used vehicles decreased.
        for(auto r = mpIndividual.routes.size(); r < params.data.nbVehicles; ++r) {
            mpIndividual.routes.emplace_back();
        }

        // Evaluate the costs of the new MP individual.
        mpIndividual.evaluateCompleteCost();

#ifndef BATCH_MODE
        std::cout << status.outPrefix << "[Decomposition] Obtained new solution of cost " << mpIndividual.cost.penalizedCost << "\n";
#endif

        if(mpIndividual.cost.penalizedCost >= eliteIndividual->cost.penalizedCost) {
            // New solution worse, or same, than elite starting one.
            // Because we keep the elite parts if the subproblems did not improve
            // on them, the new solution is never strictly worse, it can only be
            // the same as the elite one - in the worse case.
            params.stats.nSpWorsened += 1u;
        } else if(mpIndividual.cost.penalizedCost < eliteIndividual->cost.penalizedCost) {
            // New solution strictly better than elite starting one.
            params.stats.nSpImproved += 1u;
        }

        // Add the new individual to the population.
        population.addIndividualLS(&mpIndividual, true);
    }

    RouteSequenceDecomposition::SPList RouteSequenceDecomposition::getSubproblemsFromRouteIndices(const Individual& mpIndividual, const SolverStatus& status,
                                                                                                  const std::vector<int>& routeIndices) const {
        // List where we store the subproblems.
        auto sps = SPList{};

        // Number of customers in the open subproblem.
        auto currentCustomersInSP = 0;

        // Indices of routes that go in the open subproblem.
        auto currentRoutesIds = std::vector<int>{};

        for(const auto& i : routeIndices) {
            const auto& mpRoute = mpIndividual.routes.at(i);

            if(currentCustomersInSP + mpRoute.size() > params.deco.targetMaxSpCustomers) {
                // Adding the route would lead to too many customers: "close" the current SP.

                assert(!currentRoutesIds.empty());
                sps.push_back(std::make_unique<SubProblem>(params, mpIndividual, status, currentRoutesIds));

                // Reset current customers.
                currentCustomersInSP = 0;
                currentRoutesIds.clear();
            }

            // Add the current route to the list of routes going into the current SP.
            currentRoutesIds.push_back(i);
            // Increase the number of customers in the open SP accordingly.
            currentCustomersInSP += mpRoute.size();
        }

        // Check if we left the last subproblem open. In this case, close it.
        if(!currentRoutesIds.empty()) {
            sps.push_back(std::make_unique<SubProblem>(params, mpIndividual, status, currentRoutesIds));
        }

        return sps;
    }
} // namespace genvrp