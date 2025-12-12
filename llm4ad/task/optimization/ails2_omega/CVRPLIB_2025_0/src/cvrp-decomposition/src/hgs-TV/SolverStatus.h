//
// Created by alberto on 12/03/19.
//

#ifndef GENVRP_SOLVERSTATUS_H
#define GENVRP_SOLVERSTATUS_H

#include <cstdint>  // For std::uint32_t

namespace genvrp {
    /** This class represent the initial status of a solver.
     *
     *  It is used when solving a subproblem, to represent the initial status of the subproblem solver.
     *  For example, it can be used to account for the time already spent in the master problem.
     */
    struct SolverStatus {
        /** Time already elapsed when the solver is started. */
        double initialElapsedTime = 0.0;

        /** Recursion level of the subproblem.
         *
         *  The master problem is at level 0;
         *  the subproblem is at level 1;
         *  the subproblem of the subproblem is at level 2;
         *  etc.
         */
         std::uint32_t recursionLevel = 0u;

         /** Indentation prefix (deeper recursion level = deeper indentation. */
         std::string outPrefix = "";

         SolverStatus() = default;

         SolverStatus(double t, std::uint32_t r) : initialElapsedTime{t}, recursionLevel{r} {
             for(auto i = 0u; i < recursionLevel; ++i) {
                 outPrefix += "\t";
             }
         }
    };
}

#endif //GENVRP_SOLVERSTATUS_H
