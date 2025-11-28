#ifndef _FILO2_SPLITEXCHANGE_HPP_
#define _FILO2_SPLITEXCHANGE_HPP_

#include "AbstractOperator.hpp"

namespace cobra {

    class SplitExchange : public AbstractOperator {
    public:
        SplitExchange(const Instance &instance_, MoveGenerators &moves_, double tolerance_)
            : AbstractOperator(instance_, moves_, tolerance_) { }

        static constexpr bool is_symmetric = true;

    protected:
        inline void pre_processing(__attribute__((unused)) Solution &solution) override { }

        inline double compute_cost(const Solution &solution, const MoveGenerator &move) override {
            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iRoute = solution.get_route_index(i, j);
            const auto jRoute = solution.get_route_index(j, i);

            const auto iNext = solution.get_next_vertex(iRoute, i);
            const auto jNext = solution.get_next_vertex(jRoute, j);

            return -solution.get_cost_prev_vertex(iRoute, iNext) + this->instance.get_cost(i, j) -
                   solution.get_cost_prev_vertex(jRoute, jNext) + this->instance.get_cost(jNext, iNext);
        }

        bool is_feasible(Solution &solution, const MoveGenerator &move) override {
            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iRoute = solution.get_route_index(i, j);
            const auto jRoute = solution.get_route_index(j, i);


            return iRoute != jRoute && (solution.get_route_load_before_included(i) + solution.get_route_load_before_included(j) <=
                                            this->instance.get_vehicle_capacity() &&
                                        solution.get_route_load_after_included(j) - this->instance.get_demand(j) +
                                                solution.get_route_load_after_included(i) - this->instance.get_demand(i) <=
                                            this->instance.get_vehicle_capacity());
        }

        inline void execute(Solution &solution, const MoveGenerator &move, SparseIntSet &storage) override {

            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iRoute = solution.get_route_index(i, j);
            const auto jRoute = solution.get_route_index(j, i);

            assert(solution.get_first_customer(iRoute) != this->instance.get_depot());
            assert(solution.get_first_customer(jRoute) != this->instance.get_depot());

            storage.insert(this->instance.get_depot());
            for (auto curr = i; curr != this->instance.get_depot(); curr = solution.get_next_vertex(curr)) {
                storage.insert(curr);
            }

            const auto jNextNext = solution.get_next_vertex(jRoute, solution.get_next_vertex(j));
            const auto jStop = jNextNext == solution.get_first_customer(jRoute) ? this->instance.get_depot()
                                                                                : jNextNext;  // handle special case
            for (auto curr = solution.get_first_customer(jRoute); curr != jStop; curr = solution.get_next_vertex(curr)) {
                storage.insert(curr);
            }


            solution.split(i, iRoute, j, jRoute);

            if (solution.is_route_empty(iRoute)) {
                solution.remove_route(iRoute);
            }

            if (solution.is_route_empty(jRoute)) {
                solution.remove_route(jRoute);
            }
        }

        void post_processing(__attribute__((unused)) Solution &solution) override { }

        struct Cache12 {
            int v, next;
            double seqrem;
        };

        inline Cache12 prepare_cache12(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache12();
            c.v = vertex;
            c.next = solution.get_next_vertex(c.v);
            const auto route = solution.get_route_index(c.v);

            c.seqrem = -solution.get_cost_prev_vertex(route, c.next);

            return c;
        }

        inline Cache12 prepare_cache12(const Solution &solution, int vertex, int backup) {

            auto c = Cache12();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            c.next = solution.get_first_customer(route);

            c.seqrem = -solution.get_cost_prev_customer(c.next);

            return c;
        }


        inline double compute_cost(const MoveGenerator &move, const struct Cache12 i, const struct Cache12 j) {

            const auto iSequenceAdd = this->moves.get_edge_cost(move) + this->instance.get_cost(j.next, i.next);

            const auto delta = iSequenceAdd + i.seqrem + j.seqrem;

            return delta;
        }
    };

}  // namespace cobra

#endif