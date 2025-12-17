#ifndef GENVRP_ROUTESEQUENCEDECOMPOSITION_H
#define GENVRP_ROUTESEQUENCEDECOMPOSITION_H

#include "../DecompositionMethod.h"
#include "../SolverStatus.h"
#include "../SubProblem.h"

#include <vector>
#include <memory>   // For std::unique_ptr

namespace genvrp {
    /** Abstract class modelling a decomposition method that decomposes the problem in the following way.
     *  Given an individual and an ordered list of indices of routes, it scans the individual's routes
     *  in the order given by the list. It keeps putting the corresponding customers into a subproblem, until
     *  the subproblem size reaches a threshold. At that moment, it "closes" the subproblem, and starts a new one.
     */
    class RouteSequenceDecomposition : public DecompositionMethod {
    protected:
        using SPList = std::vector<std::unique_ptr<SubProblem>>;

        /** Gets a list of subproblems from the given MP individual. Any derived class must implement this. */
        virtual SPList getSubproblems(
            const Individual& mpIndividual, const SolverStatus& status) const = 0;

        /** Helper method for subclasses, to obtain subproblem just by specifying in which order the routes
         *  should be scanned. Customers are accumulated scanning the routes, until a given size is reached.
         *  At that moment, the subproblem is "closed" and a new one is opened.
         *
         *  @param mpIndividual Master problem individual to take the routes from.
         *  @param status       Solver status to pass to the subproblems.
         *  @param routeIndices Sorted sequence of the indices of routes of mpIndividual.
         *  @return             A vector of subproblems, ready to be solved.
         */
        SPList getSubproblemsFromRouteIndices(
            const Individual& mpIndividual, const SolverStatus& status,
            const std::vector<int>& routeIndices) const;

    public:
        explicit RouteSequenceDecomposition(Params& p) : DecompositionMethod{p} {}
        virtual ~RouteSequenceDecomposition() = default;

        /** Implements the abstract operator() of DecompositionMethod.
         *  It decomposes the problem, solves the subproblems, reconstructs a master problem
         *  solution from the subproblem solutions, and insert the newly built individual in
         *  the population.
         *
         *  @param population   Master problem population.
         *  @param status       Solver status given by the master problem to the subproblem.
         */
        void operator()(Population& population, const SolverStatus& status) override;
    };
}

#endif //GENVRP_ROUTESEQUENCEDECOMPOSITION_H
