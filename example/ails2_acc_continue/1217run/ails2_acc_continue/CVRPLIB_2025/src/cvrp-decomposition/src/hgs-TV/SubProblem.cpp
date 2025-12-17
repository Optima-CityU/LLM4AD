//
// Created by alberto on 13/03/19.
//

#include "SubProblem.h"
#include "Genetic.h"

#include <algorithm> // For std::max, std::min
#include <cassert>
#include <cmath>    // For std::ceil
#include <iostream> // For std::cout
#include <memory>   // For std::make_unique
#include <numeric>  // For std::iota

namespace genvrp {
    void SubProblem::initSubproblem() {
        spParams.data.nbClients = 0;
        spParams.data.totalDemand = 0.0;
        spParams.data.maxDemand = 0.0;

        // The only node which is kept for sure is the depot.
        spParams.data.cli = {mpParams.data.cli.at(0)};

        // The subproblem will have as many vehicles as there are routes.
        spParams.data.nbVehicles = routes.size();

        // The depot is always mapped to itself.
        mpToSpCustomers[0] = 0;
        spToMpCustomers[0] = 0;
    }

    void SubProblem::addRoute(int routeId) {
        assert(routeId >= 0);
        assert(routeId < mpIndividual.routes.size());

        // Get the route representation.
        const auto& route = mpIndividual.routes[routeId];

        for(const auto& customer : route) {
            // Get the Client object with info about the customer.
            const auto& client = mpParams.data.cli.at(customer);

            // Increase the number of customers in the subproblem.
            ++spParams.data.nbClients;

            // Add the 2-way mapping between SP and MP customer ids.
            mpToSpCustomers[customer] = spParams.data.nbClients;
            spToMpCustomers[spParams.data.nbClients] = customer;

            // Add the client to the subproblem.
            spParams.data.cli.push_back(client);

            // Increase the total demand in the subproblem, and update the max demand.
            spParams.data.totalDemand += client.demand;
            spParams.data.maxDemand = std::max(spParams.data.maxDemand, client.demand);
        }
    }

    void SubProblem::calculateDistances() {
        spParams.data.maxDist = 0.0;

        // Size of the distance matrix.
        const auto& sz = spParams.data.nbClients + 1;

        spParams.data.timeCost = std::vector<std::vector<double>>(sz, std::vector<double>(sz));

        // Go through the customers (and depot) added to the subproblem, and get
        // the subproblem distance matrix from the master problem one, being careful
        // that we need to use the correct mapped indices.
        for(auto i = 0; i <= spParams.data.nbClients; ++i) {
            // Distance between a vertex and itself is zero.
            spParams.data.timeCost.at(i).at(i) = 0;

            for(auto j = i + 1; j <= spParams.data.nbClients; ++j) {
                // Map the subproblem customers to the corresponding
                // master problem customers.
                const auto mpI = spToMpCustomers.at(i);
                const auto mpJ = spToMpCustomers.at(j);

                // Get the distance from the master problem.
                const auto dist = mpParams.data.timeCost.at(mpI).at(mpJ);

                // Since we deal with symmetric problems, we can just
                // update both distances at the same time.
                spParams.data.timeCost.at(i).at(j) = dist;
                spParams.data.timeCost.at(j).at(i) = dist;

                // Update the maximum distance encountered in the subproblem.
                spParams.data.maxDist = std::max(spParams.data.maxDist, dist);
            }
        }
    }

    void SubProblem::calculateProximity() {
        // Calculate the number of proximal vertices for the subproblem. This number is given by the parameter
        // nbGranular, or just by the number of customers in the subproblem if it is lower.
        const auto nProximity = std::min(spParams.ga.nbGranular, spParams.data.nbClients - 1);

        // Clear the proximal vertices list and only add an empty entry for the depot.
        spParams.data.correlatedVertices = {{}};

        // For each customer (not the depot, notice the loop starts from 1) in the subproblem...
        for(auto i = 1; i <= spParams.data.nbClients; ++i) {
            // Use this sorting criterion to sort the customers, relative to
            // customer i. The criterion is simply: if j1 is closer to i than
            // j2 is, then j1 precedes j2.
            auto distanceSort = [this, i](int j1, int j2) -> bool { return spParams.data.timeCost.at(i).at(j1) < spParams.data.timeCost.at(i).at(j2); };

            // Get a simple list of all vertices in the subproblem excluding the depot (notice that iota starts from 1).
            auto allVertices = std::vector<int>(spParams.data.nbClients);
            std::iota(allVertices.begin(), allVertices.end(), 1);

            // Sort them by proximity to customer i.
            std::sort(allVertices.begin(), allVertices.end(), distanceSort);

            // Nothing should be closer to i than i itself.
            assert(allVertices.at(0) == i);

            // Add the first nProximity customers to the proximity list for i. Notice that we start from
            // .begin() + 1 to exclude i itself, which is always the first element (see assert above).
            // Also notice that by definition of nProximity, .begin() + nProximity cannot be out of bounds:
            // at worst, it is equal to .end() when nProximity == sp_params.nb_clients and all customers
            // go into the proximity list.
            if(nProximity > 0) {
                spParams.data.correlatedVertices.emplace_back(allVertices.begin() + 1, allVertices.begin() + nProximity);
            } else {
                spParams.data.correlatedVertices.emplace_back();
            }
        }
    }

    void SubProblem::addWarmStart() {
        // Create the new individual for the subproblem, and empty it.
        spInitialIndividual = std::make_unique<Individual>(&spParams);
        spInitialIndividual->routes.clear();
        spInitialIndividual->giantTour.clear();

        for(const auto& routeNum : routes) {
            // Get the master problem route.
            const auto& route = mpIndividual.routes[routeNum];

            // Create a route in the subproblem
            spInitialIndividual->routes.emplace_back();

            // Copy the customers from the MP route to the SP route, making sure
            // the indices are mapped correctly.
            for(const auto& mpCustomer : route) {
                const auto& spCustomer = mpToSpCustomers[mpCustomer];

                spInitialIndividual->routes.back().push_back(spCustomer);
                spInitialIndividual->giantTour.push_back(spCustomer);
            }
        }

        // Evaluate the new individual.
        spInitialIndividual->evaluateCompleteCost();
    }

    SubProblem::SubProblem(genvrp::Params& mpParams, const genvrp::Individual& mpIndividual, const genvrp::SolverStatus& spStatus,
                           const std::vector<int>& routes)
        : mpParams{mpParams}, mpIndividual{mpIndividual}, spStatus{spStatus}, spParams{mpParams}, routes{routes} {
        // --- Checks that the master problem params are consistent:
        // One element of "cli" for each customer, + the depot.
        assert(mpParams.data.cli.size() == mpParams.data.nbClients + 1);
        // One element of "correlatedVertices" for each client, + the depot.
        assert(mpParams.data.correlatedVertices.size() == mpParams.data.nbClients + 1);
        // But the depot doesn't really have any correlated vertices.
        assert(mpParams.data.correlatedVertices.at(0).empty());

        // Initialise the subproblem.
        initSubproblem();

        for(const auto& routeId : routes) {
            addRoute(routeId);
        }

        // Make sure we now have the correct number of clients.
        assert(spParams.data.cli.size() == spParams.data.nbClients + 1);

        // Calculate a knapsack LB on the number of vehicles for the subproblem.
        spParams.data.lbVehicles = (int)std::ceil(spParams.data.totalDemand / spParams.data.vehicleCapacity);

        // Calculate the distances between vertices in the subproblem.
        calculateDistances();

        // Calculate the neighbouring vertices in the subproblem.
        calculateProximity();

        // Add a warm-start individual to the subproblem, based on the MP individual.
        if(mpParams.deco.warmStart) {
            addWarmStart();
        }

        // Set the subproblem termination criteria.
        spParams.ga.maxIter = mpParams.deco.spMaxIter;
        spParams.ga.maxIterNonProd = mpParams.deco.spMaxIterNonProd;

        // Set the subproblem population criteria.
        spParams.ga.nbElite = mpParams.deco.spNbElite;
        spParams.ga.lambda = mpParams.deco.spLambda;
        spParams.ga.mu = mpParams.deco.spMu;

        const auto sz = spParams.data.nbClients + 1u;

        // Reset the frequency matrices:
        if(spParams.vfreq) {
            spParams.vfreq = Params::FrequencyMatrix(sz, std::vector<std::size_t>(sz, 0u));
        }

        if(spParams.afreq) {
            spParams.afreq = Params::FrequencyMatrix(sz, std::vector<std::size_t>(sz, 0u));
        }

        if(spParams.pfreq) {
            spParams.pfreq = Params::PathFrequencyMatrix();
        }
    }

    std::optional<Individual> SubProblem::solve() {
        if(spParams.data.nbClients == 0 || spParams.data.nbVehicles == 0) {
#ifndef BATCH_MODE
            std::cout << spStatus.outPrefix << "[Subproblem] Empty subproblem (" << spParams.data.nbClients << " customers, " << spParams.data.nbVehicles
                      << " vehicles). "
                      << "Skipping.\n";
#endif

            // Empty subproblem: nothing to do.
            return std::nullopt;
        }

#ifndef BATCH_MODE
        std::cout << spStatus.outPrefix << "[Subproblem] Solving with " << spParams.data.nbClients << " customers and " << spParams.data.nbVehicles
                  << " vehicles.\n";
#endif

        // Create the solver and add the initial individual.
        Genetic solver{spParams};

        if(mpParams.deco.warmStart) {
            assert(spInitialIndividual != nullptr);
            solver.addInitialIndividual(std::move(spInitialIndividual));
        }

        // Run the Genetic Algorithm on the suproblem and return the result.
        const auto spSolution = solver.run(spStatus);

        // Add the subproblem stats to the master problem's params.
        mpParams.stats.spTime += spParams.stats.spTime;
        mpParams.stats.nSpIterations += spParams.stats.nSpIterations;

        return spSolution;
    }
} // namespace genvrp
