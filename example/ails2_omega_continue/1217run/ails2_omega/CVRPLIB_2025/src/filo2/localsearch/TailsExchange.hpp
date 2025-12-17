#ifndef _FILO2_TAILSEXCHANGE_HPP_
#define _FILO2_TAILSEXCHANGE_HPP_

#include "AbstractOperator.hpp"

namespace cobra {

    class TailsExchange : public AbstractOperator {
    public:
        TailsExchange(const Instance &instance_, MoveGenerators &moves_, double tolerance_)
            : AbstractOperator(instance_, moves_, tolerance_) { }

        static constexpr bool is_symmetric = false;

    protected:
        inline void pre_processing(__attribute__((unused)) Solution &solution) override { }

        inline double compute_cost(const Solution &solution, const MoveGenerator &move) override {
            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iRoute = solution.get_route_index(i, j);
            const auto jRoute = solution.get_route_index(j, i);

            const auto iNext = solution.get_next_vertex(iRoute, i);
            const auto jPrev = solution.get_prev_vertex(jRoute, j);

            const auto delta = -solution.get_cost_prev_vertex(iRoute, iNext) + this->instance.get_cost(i, j) -
                               solution.get_cost_prev_vertex(jRoute, j) + this->instance.get_cost(jPrev, iNext);

            return delta;
        }

        bool is_feasible(Solution &solution, const MoveGenerator &move) override {
            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iRoute = solution.get_route_index(i, j);
            const auto jRoute = solution.get_route_index(j, i);


            return iRoute != jRoute &&
                   // iRoute != jRoute if both i and j are different from the depot
                   solution.get_route_load_before_included(i) + solution.get_route_load_after_included(j) <=
                       this->instance.get_vehicle_capacity() &&
                   solution.get_route_load_before_included(j) - this->instance.get_demand(j) + solution.get_route_load_after_included(i) -
                           this->instance.get_demand(i) <=
                       this->instance.get_vehicle_capacity();
        }

        inline void execute(Solution &solution, const MoveGenerator &move, SparseIntSet &storage) override {

            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iNext = solution.get_next_vertex(i);
            const auto jPrev = solution.get_prev_vertex(j);

            const auto i_route = solution.get_route_index(i);
            const auto j_route = solution.get_route_index(j);

            storage.insert(i);
            storage.insert(iNext);
            storage.insert(jPrev);
            storage.insert(j);

            this->update_bits.at(i, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iNext, UPDATE_BITS_SECOND, true);
            this->update_bits.at(j, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jPrev, UPDATE_BITS_FIRST, true);

            solution.swap_tails(i, i_route, j, j_route);

            if (solution.is_route_empty(i_route)) {
                solution.remove_route(i_route);
            }

            if (solution.is_route_empty(j_route)) {
                solution.remove_route(j_route);
            }
        }

        void post_processing(__attribute__((unused)) Solution &solution) override { }

        struct Cache12 {
            int v, next, prev;
            double seq1rem, seq2rem;
        };

        inline Cache12 prepare_cache12(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache12();
            c.v = vertex;
            const auto route = solution.get_route_index(c.v);
            c.next = solution.get_next_vertex(c.v);
            c.prev = solution.get_prev_vertex(c.v);
            c.seq1rem = -solution.get_cost_prev_vertex(route, c.next);
            c.seq2rem = -solution.get_cost_prev_customer(c.v);

            return c;
        }

        inline Cache12 prepare_cache12(const Solution &solution, int vertex, int backup) {

            auto c = Cache12();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            c.next = solution.get_first_customer(route);
            c.prev = solution.get_last_customer(route);
            c.seq1rem = -solution.get_cost_prev_customer(c.next);
            c.seq2rem = -solution.get_cost_prev_depot(route);

            return c;
        }

        inline std::pair<double, double> compute_cost_pair(const MoveGenerator &move, const struct Cache12 i, const struct Cache12 j) {

            const auto c_iv_jv = this->moves.get_edge_cost(move);

            const auto delta1 = i.seq1rem + c_iv_jv + j.seq2rem + this->instance.get_cost(j.prev, i.next);
            const auto delta2 = j.seq1rem + c_iv_jv + i.seq2rem + this->instance.get_cost(i.prev, j.next);

            return {delta1, delta2};
        }

        struct Cache1 {
            int v, next;
            double seq1rem;
        };

        inline Cache1 prepare_cache1(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache1();
            c.v = vertex;
            c.next = solution.get_next_vertex(c.v);
            const auto route = solution.get_route_index(c.v);
            c.seq1rem = -solution.get_cost_prev_vertex(route, c.next);

            return c;
        }

        inline Cache1 prepare_cache1(const Solution &solution, int vertex, int backup) {
            auto c = Cache1();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            c.next = solution.get_first_customer(route);
            c.seq1rem = -solution.get_cost_prev_customer(c.next);

            return c;
        }

        struct Cache2 {
            int v, prev;
            double seq2rem;
        };

        inline Cache2 prepare_cache2(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache2();
            c.v = vertex;
            c.prev = solution.get_prev_vertex(c.v);
            c.seq2rem = -solution.get_cost_prev_customer(c.v);

            return c;
        }

        inline Cache2 prepare_cache2(const Solution &solution, int vertex, int backup) {
            auto c = Cache2();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            c.prev = solution.get_last_customer(route);
            c.seq2rem = -solution.get_cost_prev_depot(route);

            return c;
        }

        template <typename C1, typename C2>
        inline double compute_cost(const MoveGenerator &move, const C1 i, const C2 j) {

            const auto c_iv_jv = this->moves.get_edge_cost(move);

            const auto delta1 = i.seq1rem + c_iv_jv + j.seq2rem + this->instance.get_cost(j.prev, i.next);

            return delta1;
        }
    };

}  // namespace cobra
#endif