//
// Created by alberto on 13/03/19.
//

#ifndef GENVRP_SUBPROBLEM_H
#define GENVRP_SUBPROBLEM_H

#include "Params.h"
#include "Individual.h"
#include "SolverStatus.h"

#include <map>
#include <vector>
#include <optional>
#include <memory>   // For std::unique_ptr

namespace genvrp {
    /** A subproblem is a smaller problem obtained from a master problem via decomposition. */
    class SubProblem {
        /** Master problem params. */
        Params& mpParams;

        /** Master problem individual used as a base for the subproblem.
         *
         *  In most cases, this means that the customers in the subproblem correspond to customers
         *  visited in some routes of this individual.
         */
        const Individual& mpIndividual;

    public:

        /** Initial status of the subproblem solver.
         *
         *  This can be used, for example, to specify that when the genetic
         *  algorithm starts on the subproblem, some time has already elapsed,
         *  so the timeout should be calculated accordingly.
         */
        const SolverStatus& spStatus;

    private:

        /** Suproblem params. */
        Params spParams;

        /** An individual to put in the initial population of the subproblem, generated based on the
         *  master problem individual associated with this subproblem.
         */
        std::unique_ptr<Individual> spInitialIndividual;

        /** Mapping of customer indices from the master problem indices to
         *  the subproblem indices.
         *
         *  Customer indexed by i in the master problem, is mpToSpCustomers[i]
         *  in the subproblem (assuming this customer was included in the
         *  subproblem at all).
         */
        std::map<int, int> mpToSpCustomers;

    public:

        /** Mapping of customer indices from the subproblem indices to the
         *  master problem indices.
         *
         *  Customer indexed by i in the subproblem, was spToMpCustomers[i]
         *  in the master problem.
         */
        std::map<int, int> spToMpCustomers;

        /** Index of the master problem individual's routes used to build the subproblem. */
        std::vector<int> routes;

    private:

        /** Initialises an empty subproblem, given the routes to use. */
        void initSubproblem();

        /** Add customers from one route, given its index in mpIndividual. */
        void addRoute(int routeId);

        /** Recomputes the distance matrix in the subproblem. */
        void calculateDistances();

        /** Recomputes the neighbouring vertices in the subproblem. */
        void calculateProximity();

        /** Creates a warm-start individual for the subproblem, based on the starting master problem individual. */
        void addWarmStart();

    public:

        /** Builds a subproblem, given the master problem params; the starting individual; the subproblem status;
         *  and a list of route indices corresponding to the mpIndividual routes to include in the subproblem.
         */
        SubProblem(
            Params& mpParams,
            const Individual& mpIndividual,
            const SolverStatus& spStatus,
            const std::vector<int>& routes);

        /** Solve the subproblem and return the best individual found, if any. */
        std::optional<Individual> solve();
};
}

#endif //GENVRP_SUBPROBLEM_H
