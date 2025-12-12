#ifndef GENVRP_DECOMPOSITIONMETHOD_H
#define GENVRP_DECOMPOSITIONMETHOD_H

#include "Params.h"
#include "Population.h"
#include "SubProblem.h"
#include "SolverStatus.h"

#include <memory>   // For std::unique_ptr
#include <optional>

namespace genvrp {
    /** Abstract base class for decomposition methods. */
    class DecompositionMethod {
    protected:
        /** Master problem parameters/ */
        Params& params;

        /** Master problem elite individual used to decompose. */
        std::optional<Individual> eliteIndividual;

        /** Takes a subproblem individual (spSolution) and adds its partial solution to the master
         *  problem individual mpIndividual.
         *
         *  @param spSolution       A solution to the subproblem.
         *  @param mpIndividual     The master problem individual being built.
         *  @param subproblem       The subproblem which generated spSolution.
         */
        void mergeSpSolutionIntoMpIndividual(
            const Individual& spSolution, Individual& mpIndividual, const SubProblem& subproblem) const;

    public:

        /** Decompose the problem, solve the subproblems, create a new solution, and insert it in the population.
         *
         *  @param  population  Master problem population.
         *  @param  status      Solver status given from the master problem to the subproblem.
         */
        virtual void operator()(Population& population, const SolverStatus& status) = 0;

        /** Construct from the master problem's parameters. */
        explicit DecompositionMethod(Params& p) : params{p} {}

        /** A virtual destructor is required for an abstract class. */
        virtual ~DecompositionMethod() = default;
    };

    /** Returns a pointer to a concrete decomposition method.
     *
     *  The method is chosen by looking at Params::decompositionType.
     *
     *  @param params   Problem parameters.
     *  @return         A pointer to a concrete decomposition method.
     */
    std::unique_ptr<DecompositionMethod> getDecompositionMethod(Params& params);
}

#endif //GENVRP_DECOMPOSITIONMETHOD_H
