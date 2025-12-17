#include "Genetic.h"

#include <chrono>
#include <ctime>
#include <iostream>
#include <limits>
#include <random>

namespace genvrp {
    std::optional<Individual> Genetic::run(const SolverStatus& initialStatus) {
        using namespace std::chrono;

        // Reset the elapsed time.
        elapsed = 0.0;

#ifndef BATCH_MODE
        std::cout << initialStatus.outPrefix << "[Algorithm] Starting the genetic algorithm.\n";
#endif

        // Cannot start with negative elapsed time!
        assert(initialStatus.initialElapsedTime >= 0);

		// Number of consecutive iterations without improvement.
        int nbIterNonProd = 1;

        // Current iteration number.
        int nbIter = 0;

        // CPU time start of the algorithm (using std::clock for CPU time).
        const std::clock_t startTime = std::clock();
        std::clock_t lastDecoTime = startTime;

		// While the iteration termination criteria are not met:
        while(nbIter < params.ga.maxIter && nbIterNonProd < params.ga.maxIterNonProd) {
            // Check if we are in timeout (using CPU time):
            const std::clock_t currentTime = std::clock();
            elapsed = static_cast<double>(currentTime - startTime) / CLOCKS_PER_SEC;

			// In this case, exit.
            if(elapsed > params.ga.timeoutSec - initialStatus.initialElapsedTime) {
                break;
            }

            // Generate a new individual (offspring) via cross-over.
            auto offspring = crossoverOX(*population.getBinaryTournament(), *population.getBinaryTournament());

            // Run local search on the new individual.
            localSearch.run(&offspring, params.ga.penaltyCapacity, params.ga.penaltyDuration);

			// Update the frequency table, if it is being used.
            if(params.vfreq) {
                updateCustomerFrequencyMatrix(offspring);
            }

            // Update the arc frequency matrix, if it is being used.
            if(params.afreq) {
                updateArcFrequencyMatrix(offspring);
            }

            // Update the path frequency matrix, if it is being used.
            if(params.pfreq) {
                updatePathFrequencyMatrix(offspring);
            }

			// Add the new individual to the population.
            bool isNewBest = population.addIndividual(&offspring, true);

			// If the individual is unfeasible, try to repair it with
            // a certain probability.
            if(!offspring.isFeasible && std::rand() % 2 == 0) {
                localSearch.run(&offspring, params.ga.penaltyCapacity * 10., params.ga.penaltyDuration * 10.);

				// If the repair was succesfull, also add the repaired
                // individual to the population.
                if(offspring.isFeasible)
                    isNewBest = (population.addIndividual(&offspring, false) || isNewBest);
            }

            
            // If the best solution was improved by the new individual...
            if(isNewBest) {
                // Reset the counter
                nbIterNonProd = 1;

                // Update the stats about new best solutions.
                if(params.stats.recordBestSolutionUpdates) {
                    const std::clock_t timeAtUpdate = std::clock();
                    const double elapsedAtUpdate = static_cast<double>(timeAtUpdate - startTime) / CLOCKS_PER_SEC;

                    params.stats.newBestIndividuals.push_back(
                        {Params::RunStats::NewBestSource::MainGenetic, offspring.cost.penalizedCost, elapsedAtUpdate});
                }
            } else {
                nbIterNonProd++;
            }

            // Adjust the penalty multipliers.
            if(nbIter % params.ga.adjMultiplierIters == 0) {
                population.managePenalties();
            }

// Print info on the current state of the search.
#ifndef BATCH_MODE
            if(nbIter % params.ga.outputIters == 0) {
                population.printState(nbIter, nbIterNonProd, initialStatus.outPrefix);
            }
#endif

            // Decompose the problem.
            if(decomposition && nbIter > 0 && nbIter % params.deco.decompositionIters == 0) {
#ifndef BATCH_MODE
                std::cout << initialStatus.outPrefix << "[Algorithm] Attempting decomposition.\n";
#endif

                // If we are in the master problem, record its runtime.
                if(initialStatus.recursionLevel == 0u) {
                    const std::clock_t decoTime = std::clock();
                    const double mpTimeSinceDeco = static_cast<double>(decoTime - lastDecoTime) / CLOCKS_PER_SEC;
                    params.stats.mpTime += mpTimeSinceDeco;
                    params.stats.nMpDecompositions += 1u;
                }

                // Pass the solver status made of the current elapsed time
                // and the recursion level, which is the current recursion
                // level, increased by 1.
                (*decomposition)(population, SolverStatus{elapsed, initialStatus.recursionLevel + 1u});

                lastDecoTime = std::clock();
            }

            // If we reached the termination criterion in the main problem but there is some time left, I suggest restarting the algorithm to profit from the remaining time
            if(initialStatus.recursionLevel == 0u && (nbIter == params.ga.maxIter || nbIterNonProd == params.ga.maxIterNonProd)) {
                population.restart();
                nbIterNonProd = 1;
            }

			// Increase the current iteration number.
            ++nbIter;
        }

        // Calculate the elapsed CPU time.
        const std::clock_t finalTime = std::clock();
        elapsed = static_cast<double>(finalTime - startTime) / CLOCKS_PER_SEC;

        // Because it gets rounded up, check it is not more than the timeout.
        elapsed = std::min(elapsed, params.ga.timeoutSec);

        // If we are in the masterproblem, record its parameters.
        if(initialStatus.recursionLevel == 0u) {
            params.stats.mpTime += static_cast<double>(finalTime - lastDecoTime) / CLOCKS_PER_SEC;
            params.stats.nMpIterations = nbIter;
        }

        // If we are in a subproblem, record its parameters.
        if(initialStatus.recursionLevel > 0u) {
            params.stats.spTime = elapsed;
            params.stats.nSpIterations = nbIter;
        }

#ifndef BATCH_MODE
        std::cout << initialStatus.outPrefix << "[Algorithm] Recursion level: " << initialStatus.recursionLevel << "\n";
        std::cout << initialStatus.outPrefix << "[Algorithm] Time elapsed: " << elapsed << " seconds\n";
        std::cout << initialStatus.outPrefix << "[Algorithm] Number of iterations: " << nbIter << "\n";
#endif

        // Get the best feasible individual, if any.
        std::optional<const Individual*> bestIndiv = population.getBestFound();
        if(!bestIndiv) {
            // Otherwise, get the best infeasible one, if any.
            bestIndiv = population.getBestInfeasible();
        }
        if(!bestIndiv) {
            // No feasible and no infeasible individuals found.
            return std::nullopt;
        }

        // Return a copy of the best individual.
        return **bestIndiv;
    }

    Individual Genetic::crossoverOX(const Individual& parent1, const Individual& parent2) {
        // We initialize a frequency table to track the customers which have been already inserted.
        auto freqClient = std::vector<bool>(params.data.nbClients + 1, false);

        // We pick the beginning and end of the crossover zone.
        int start = std::rand() % params.data.nbClients;
        int end = std::rand() % params.data.nbClients;

        // Avoid that start and end coincide by accident.
        while(end == start && params.data.nbClients > 1) {
            end = std::rand() % params.data.nbClients;
        }

        // Create the resulting individual.
        Individual result{&params};

        int j = start;
        // We keep the elements from "start" to "end".
        while((j % params.data.nbClients) != ((end + 1) % params.data.nbClients)) {
            result.giantTour[j % params.data.nbClients] = parent1.giantTour[j % params.data.nbClients];
            freqClient[result.giantTour[j % params.data.nbClients]] = true;
            j++;
        }

        // We fill the rest of the elements in the order of the second parent.
        for(int i = 1; i <= params.data.nbClients; i++) {
            int index = parent2.giantTour[(end + i) % params.data.nbClients];
            if(freqClient[index] == false) {
                result.giantTour[j % params.data.nbClients] = index;
                j++;
            }
        }

        // Run the individual through the split algorithm.
        split.generalSplit(&result, parent1.cost.nbRoutes);

        // Any decent compiler will implement return-value optimisation
        // (copy elision) and will move "result" at the caller site, thus
        // avoiding a copy.
        return result;
    }

    void Genetic::updateCustomerFrequencyMatrix(const Individual& solution) const {
        assert(params.vfreq);

        for(const auto& route : solution.routes) {
            for(auto i = 0u; i < route.size(); ++i) {
                for(auto j = i + 1; j < route.size(); ++j) {
                    (*params.vfreq)[route[i]][route[j]] += 1;
                    (*params.vfreq)[route[j]][route[i]] += 1;
                }
            }
        }
    }

    void Genetic::updateArcFrequencyMatrix(const Individual& solution) const {
        assert(params.afreq);

        for(const auto& route : solution.routes) {
            if(route.empty()) {
                continue;
            }

            for(auto i = 0u; i < route.size() - 1u; ++i) {
                assert(route[i] != 0);
                assert(route[i + 1] != 0);

                (*params.afreq)[route[i]][route[i + 1]] += 1;
            }
        }
    }

    void Genetic::updatePathFrequencyMatrix(const genvrp::Individual& solution) const {
        assert(params.pfreq);

        for(const auto& route : solution.routes) {
            if(route.empty()) {
                continue;
            }

            for(auto it = route.begin(); it <= route.end() - Params::record_path_size; ++it) {
                const auto path = std::vector<int>(it, it + Params::record_path_size);
                const auto trieIt = params.pfreq->find(path);

                if(trieIt != params.pfreq->end()) {
                    ++(trieIt->second);
                } else {
                    (*params.pfreq)[path] = 1u;
                }
            }
        }
    }
} // namespace genvrp