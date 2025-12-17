#ifndef GENVRP_ARCBASEDDECOMPOSITION_H
#define GENVRP_ARCBASEDDECOMPOSITION_H

#include <vector>
#include <iostream>
#include <utility>  // For std::pair

#include "../DecompositionMethod.h"
#include "../SolverStatus.h"
#include "../SubProblem.h"

namespace genvrp {
    /** Abstract class that models decomposition in the following way. Given an elite individual and a set
     *  of arcs used by the solution encoded in that individual, it fixes those arcs and leaves the remaining
     *  ones free for optimisation in the subproblem.
     */
    class ArcBasedDecomposition : public DecompositionMethod {
    public:
        /** Easiest possible struct to identify an arc. */
        struct Arc {
            int origin;
            int destination;
            double cost;

            Arc(int o, int d, double c = 0.0) : origin{o}, destination{d}, cost{c} {}

            bool operator==(const Arc& other) const {
                return origin == other.origin && destination == other.destination;
            }
            bool operator<(const Arc& other) const { // Only needed to put arcs in a set.
                return std::tie(origin, destination) < std::tie(other.origin, other.destination);
            }
        };

        /** A path as a sequence of consecutive vertices. */
        using Path = std::vector<int>;

        /** A list of customers. */
        using CustomerSet = std::vector<int>;

    protected:

        /** This method is used to obtain a sorted list of arcs to fix. These will be gradually
         *  merged until the required subproblem size is obtained. */
        virtual std::vector<Arc> getArcsToFix(const Individual& mpIndividual) const = 0;

        /** Given a set of arcs, checks if some of them are consecutive. In this case, they form a path.
         *
         *  This method returns all such paths (isolated arcs are considered as 2-customer paths) as the second
         *  member of the pair. The first member of the pair is just the list of all customers visited by the
         *  given arcs.
         */
        std::pair<CustomerSet, std::vector<Path>> getPaths(const std::vector<Arc>& arcs) const;

    public:
        /** Implements the abstract operator() of DecompositionMethod.
         *  It fixes some arcs, solve the resulting subproblem, reconstructs a master problem
         *  solution from the subproblem solution, and insert the newly built individual in
         *  the population.
         *
         *  @param population   Master problem population.
         *  @param status       Solver status given by the master problem to the subproblem.
         */
        void operator()(Population& population, const SolverStatus& status) override;

        explicit ArcBasedDecomposition(Params& p) : DecompositionMethod{p} {}
        virtual ~ArcBasedDecomposition() = default;
    };
}

#endif //GENVRP_ARCBASEDDECOMPOSITION_H
