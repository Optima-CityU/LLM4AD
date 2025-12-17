#ifndef GENETIC_H
#define GENETIC_H

#include "Population.h"
#include "Individual.h"
#include "SolverStatus.h"
#include "DecompositionMethod.h"

#include <optional>
#include <memory>   // For std::unique_ptr

namespace genvrp {
    /** This class models a Genetic Algorithm solver. */
    class Genetic {
        /** Reference to the problem parameters and data. */
        Params& params;

        /** Reference to the splitter object, used to optimally obtain routes from a giant-tour representation. */
        Split split;

        /** Reference to the local search object, used to improve an individual's fitness. */
        LocalSearch localSearch;

        /** Reference to the population, i.e., the set of feasible and unfeasible individuals used by the G.A. */
        Population population;

        /** Pointer to the decomposition method. It can be null if no decomposition is active. */
        std::unique_ptr<DecompositionMethod> decomposition;

        /** Elapsed time in seconds. */
        double elapsed = 0.0;

        /** Applies the OX cross-over operator to two parent individual, to produce a child individual.
         *
         *  @param parent1  First parent individual.
         *  @param parent2  Second parent individual.
         *  @return         Child individual produced by the operator.
         */
        Individual crossoverOX(const Individual& parent1, const Individual& parent2);

        /** Updates the customers frequency matrix keeping track of which clients appear together in a same route.
         *
         *  @param solution The individual whose routes we consider in the update.
         */
        void updateCustomerFrequencyMatrix(const Individual& solution) const;

        /** Updates the arc frequency matrix keeping track of which arcs are used often in good routes.
         *
         *  @param solution The individual whose routes we consider in the update.
         */
        void updateArcFrequencyMatrix(const Individual& solution) const;

        /** Updates the path frequency matrix keeping track of which paths are used often in good routes.
         *
         *  @param  solution The individual whose routes we consider in the update.
         */
        void updatePathFrequencyMatrix(const Individual& solution) const;

        /** Updates the historical customers frequency matrix keeping track of which clients appear together in
         *  a same route.
         *
         *  @param solution The individual whose routes we consider in the update.
         *  @param nbIter   Current iteration number.
         */
        void updateHistoricalCustomerFrequencyMatrix(const Individual& solution, int nbIter) const;

    public:

        void printBestSolutionValue() const;

        /** Runs the genetic algorithm and prints relevant statistics to stdout.
         *
         *  @param initialStatus    Initial status of the solver telling, e.g., how much time elapsed
         *                          before the solver starts.
         *  @return                 The best feasible individual, if any. Otherwise, the best unfeasible one,
         *                          if any. Otherwise, nullopt.
         */
        std::optional<Individual> run(const SolverStatus& initialStatus = SolverStatus{});

        /** Adds an initial individual to the population. */
        void addInitialIndividual(std::unique_ptr<Individual> initial) { population.addIndividual(&*initial, false); }

        /** Writes the best feasible solution to file. */
        void exportBestSolution() {
            if(population.getBestFound() != NULL) {
                population.getBestFound()->exportCVRPLibFormat(params.pathToSolution);
            } else {
                std::cout << "[Algorithm] No feasible solution produced.\n";
            }
        }

        /** Constructs a Genetic Algorithm solver.
         *
         *  @param params       Problem parameters and data.
         *  @param split        Splitter object, to split giant tours into routes.
         *  @param population   Population of feasible and infeasible individuals.
         *  @param localSearch  Local search to improve (or repair) individuals.
         *  @param ticks        Maximum runtime of the algorithm.
         */
        Genetic(Params& params) :
            params{params},
              split{this->params},
              localSearch{this->params},
              population{this->params, this->split, this->localSearch},
              decomposition{getDecompositionMethod(this->params)} {}
    };
}
#endif
