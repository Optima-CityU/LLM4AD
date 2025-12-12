#ifndef _FILO2_THREETHREEEXCHANGE_HPP_
#define _FILO2_THREETHREEEXCHANGE_HPP_

#include "AbstractOperator.hpp"

namespace cobra {

    class ThreeThreeExchange : public AbstractOperator {
    public:
        ThreeThreeExchange(const Instance &instance_, MoveGenerators &moves_, double tolerance_)
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
            const auto iPrev = solution.get_prev_vertex(iRoute, i);
            const auto iPrevPrev = solution.get_prev_vertex(iRoute, iPrev);
            const auto iPrevPrevPrev = solution.get_prev_vertex(iRoute, iPrevPrev);

            const auto jPrev = solution.get_prev_vertex(jRoute, j);
            const auto jPrevPrev = solution.get_prev_vertex(jRoute, jPrev);
            const auto jPrevPrevPrev = solution.get_prev_vertex(jRoute, jPrevPrev);
            const auto jPrevPrevPrevPrev = solution.get_prev_vertex(jRoute, jPrevPrevPrev);

            const auto iSequenceRem = -solution.get_cost_prev_vertex(iRoute, iPrevPrev) - solution.get_cost_prev_vertex(iRoute, iNext);
            const auto jSequenceRem = -solution.get_cost_prev_vertex(jRoute, jPrevPrevPrev) - solution.get_cost_prev_vertex(jRoute, j);

            const auto iSequenceAdd = +this->instance.get_cost(jPrevPrevPrevPrev, iPrevPrev) + this->instance.get_cost(i, j);
            const auto jSequenceAdd = +this->instance.get_cost(iPrevPrevPrev, jPrevPrevPrev) + this->instance.get_cost(jPrev, iNext);

            const auto delta = iSequenceAdd + jSequenceAdd + iSequenceRem + jSequenceRem;

            return delta;
        }

        bool is_feasible(Solution &solution, const MoveGenerator &move) override {
            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iRoute = solution.get_route_index(i, j);
            const auto jRoute = solution.get_route_index(j, i);

            const auto iPrev = solution.get_prev_vertex(iRoute, i);
            const auto iPrevPrev = solution.get_prev_vertex(iRoute, iPrev);

            const auto jPrev = solution.get_prev_vertex(jRoute, j);
            const auto jPrevPrev = solution.get_prev_vertex(jRoute, jPrev);
            const auto jPrevPrevPrev = solution.get_prev_vertex(jRoute, jPrevPrev);

            return (iRoute != jRoute && iPrev != this->instance.get_depot() && iPrevPrev != this->instance.get_depot() &&
                    jPrev != this->instance.get_depot() && jPrevPrev != this->instance.get_depot() &&
                    jPrevPrevPrev != this->instance.get_depot() &&
                    solution.get_route_load(jRoute) - this->instance.get_demand(jPrev) - this->instance.get_demand(jPrevPrev) -
                            this->instance.get_demand(jPrevPrevPrev) + this->instance.get_demand(i) + this->instance.get_demand(iPrev) +
                            this->instance.get_demand(iPrevPrev) <=
                        this->instance.get_vehicle_capacity() &&
                    solution.get_route_load(iRoute) + this->instance.get_demand(jPrev) + this->instance.get_demand(jPrevPrev) +
                            this->instance.get_demand(jPrevPrevPrev) - this->instance.get_demand(i) - this->instance.get_demand(iPrev) -
                            this->instance.get_demand(iPrevPrev) <=
                        this->instance.get_vehicle_capacity()) ||
                   (iRoute == jRoute && i != jPrev && i != jPrevPrev && i != jPrevPrevPrev &&
                    solution.get_next_vertex(iRoute, i) != jPrevPrevPrev && j != iPrev && j != iPrevPrev);
        }

        inline void execute(Solution &solution, const MoveGenerator &move, SparseIntSet &storage) override {

            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iRoute = solution.get_route_index(i, j);
            const auto jRoute = solution.get_route_index(j, i);

            const auto iPrev = solution.get_prev_vertex(iRoute, i);
            const auto iPrevPrev = solution.get_prev_vertex(iRoute, iPrev);
            const auto iPrevPrevPrev = solution.get_prev_vertex(iRoute, iPrevPrev);

            const auto iNext = solution.get_next_vertex(iRoute, i);
            const auto iNextNext = solution.get_next_vertex(iRoute, iNext);
            const auto iNextNextNext = solution.get_next_vertex(iRoute, iNextNext);
            const auto iNextNextNextNext = solution.get_next_vertex(iRoute, iNextNextNext);

            const auto jPrev = solution.get_prev_vertex(jRoute, j);
            const auto jPrevPrev = solution.get_prev_vertex(jRoute, jPrev);
            const auto jPrevPrevPrev = solution.get_prev_vertex(jRoute, jPrevPrev);
            const auto jPrevPrevPrevPrev = solution.get_prev_vertex(jRoute, jPrevPrevPrev);

            const auto jNext = solution.get_next_vertex(jRoute, j);
            const auto jNextNext = solution.get_next_vertex(jRoute, jNext);
            const auto jNextNextNext = solution.get_next_vertex(jRoute, jNextNext);


            storage.insert(iPrevPrevPrev);
            storage.insert(iPrevPrev);
            storage.insert(iPrev);
            storage.insert(i);
            storage.insert(iNext);
            storage.insert(iNextNext);
            storage.insert(iNextNextNext);
            storage.insert(iNextNextNextNext);
            storage.insert(jPrevPrevPrevPrev);
            storage.insert(jPrevPrevPrev);
            storage.insert(jPrevPrev);
            storage.insert(jPrev);
            storage.insert(j);
            storage.insert(jNext);
            storage.insert(jNextNext);
            storage.insert(jNextNextNext);

            this->update_bits.at(iPrevPrevPrev, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iPrevPrev, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iPrevPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iPrev, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(i, UPDATE_BITS_FIRST, true);
            this->update_bits.at(i, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iNext, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iNextNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iNextNext, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iNextNextNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iNextNextNext, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iNextNextNextNext, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jPrevPrevPrevPrev, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jPrevPrevPrev, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jPrevPrevPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jPrevPrev, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jPrevPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jPrev, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(j, UPDATE_BITS_FIRST, true);
            this->update_bits.at(j, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jNext, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jNextNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jNextNext, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jNextNextNext, UPDATE_BITS_SECOND, true);


            solution.remove_vertex(iRoute, i);
            solution.remove_vertex(iRoute, iPrev);
            solution.remove_vertex(iRoute, iPrevPrev);

            solution.insert_vertex_before(jRoute, j, iPrevPrev);
            solution.insert_vertex_before(jRoute, j, iPrev);
            solution.insert_vertex_before(jRoute, j, i);

            solution.remove_vertex(jRoute, jPrev);
            solution.remove_vertex(jRoute, jPrevPrev);
            solution.remove_vertex(jRoute, jPrevPrevPrev);

            solution.insert_vertex_before(iRoute, iNext, jPrevPrevPrev);
            solution.insert_vertex_before(iRoute, iNext, jPrevPrev);
            solution.insert_vertex_before(iRoute, iNext, jPrev);
        }

        void post_processing(__attribute__((unused)) Solution &solution) override { }

        struct Cache12 {
            int v, prev, prevprev, prevprevprev, prevprevprevprev, next;
            double seqrem1, seqrem2;
        };

        inline Cache12 prepare_cache12(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache12();
            c.v = vertex;
            const auto route = solution.get_route_index(c.v);
            c.prev = solution.get_prev_vertex(c.v);
            c.prevprev = solution.get_prev_vertex(route, c.prev);
            c.prevprevprev = solution.get_prev_vertex(route, c.prevprev);
            c.prevprevprevprev = solution.get_prev_vertex(route, c.prevprevprev);
            c.next = solution.get_next_vertex(c.v);

            c.seqrem1 = -solution.get_cost_prev_vertex(route, c.prevprev) - solution.get_cost_prev_vertex(route, c.next);
            c.seqrem2 = -solution.get_cost_prev_vertex(route, c.prevprevprev) - solution.get_cost_prev_customer(c.v);

            return c;
        }


        inline Cache12 prepare_cache12(const Solution &solution, int vertex, int backup) {

            auto c = Cache12();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            c.prev = solution.get_last_customer(route);
            c.prevprev = solution.get_prev_vertex(c.prev);
            c.prevprevprev = solution.get_prev_vertex(route, c.prevprev);
            c.prevprevprevprev = solution.get_prev_vertex(route, c.prevprevprev);
            c.next = solution.get_first_customer(route);

            c.seqrem1 = -solution.get_cost_prev_vertex(route, c.prevprev) - solution.get_cost_prev_customer(c.next);
            c.seqrem2 = -solution.get_cost_prev_vertex(route, c.prevprevprev) - solution.get_cost_prev_depot(route);

            return c;
        }

        inline std::pair<double, double> compute_cost_pair(const MoveGenerator &move, const struct Cache12 i, const struct Cache12 j) {

            const auto c_iv_jv = this->moves.get_edge_cost(move);

            const auto c_iprevprevprev_jprevprevprev = this->instance.get_cost(i.prevprevprev, j.prevprevprev);

            const auto delta1 = this->instance.get_cost(j.prevprevprevprev, i.prevprev) + c_iv_jv + c_iprevprevprev_jprevprevprev +
                                this->instance.get_cost(j.prev, i.next) + i.seqrem1 + j.seqrem2;
            const auto delta2 = this->instance.get_cost(i.prevprevprevprev, j.prevprev) + c_iv_jv + c_iprevprevprev_jprevprevprev +
                                this->instance.get_cost(i.prev, j.next) + j.seqrem1 + i.seqrem2;

            return {delta1, delta2};
        }

        struct Cache1 {
            int v, prevprev, prevprevprev, next;
            double seqrem1;
        };

        inline Cache1 prepare_cache1(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache1();
            c.v = vertex;
            const auto route = solution.get_route_index(c.v);
            const auto prev = solution.get_prev_vertex(c.v);
            c.prevprev = solution.get_prev_vertex(route, prev);
            c.prevprevprev = solution.get_prev_vertex(route, c.prevprev);
            c.next = solution.get_next_vertex(c.v);

            c.seqrem1 = -solution.get_cost_prev_vertex(route, c.prevprev) - solution.get_cost_prev_vertex(route, c.next);

            return c;
        }

        inline Cache1 prepare_cache1(const Solution &solution, int vertex, int backup) {
            auto c = Cache1();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            const auto prev = solution.get_last_customer(route);
            c.prevprev = solution.get_prev_vertex(prev);
            c.prevprevprev = solution.get_prev_vertex(route, c.prevprev);
            c.next = solution.get_first_customer(route);

            c.seqrem1 = -solution.get_cost_prev_vertex(route, c.prevprev) - solution.get_cost_prev_customer(c.next);

            return c;
        }

        struct Cache2 {
            int v, prev, prevprevprev, prevprevprevprev;
            double seqrem2;
        };

        inline Cache2 prepare_cache2(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache2();
            c.v = vertex;
            const auto route = solution.get_route_index(c.v);
            c.prev = solution.get_prev_vertex(c.v);
            const auto prevprev = solution.get_prev_vertex(route, c.prev);
            c.prevprevprev = solution.get_prev_vertex(route, prevprev);
            c.prevprevprevprev = solution.get_prev_vertex(route, c.prevprevprev);

            c.seqrem2 = -solution.get_cost_prev_vertex(route, c.prevprevprev) - solution.get_cost_prev_customer(c.v);

            return c;
        }

        inline Cache2 prepare_cache2(const Solution &solution, int vertex, int backup) {
            auto c = Cache2();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            c.prev = solution.get_last_customer(route);
            const auto prevprev = solution.get_prev_vertex(c.prev);
            c.prevprevprev = solution.get_prev_vertex(route, prevprev);
            c.prevprevprevprev = solution.get_prev_vertex(route, c.prevprevprev);

            c.seqrem2 = -solution.get_cost_prev_vertex(route, c.prevprevprev) - solution.get_cost_prev_depot(route);

            return c;
        }

        template <typename C1, typename C2>
        inline double compute_cost(const MoveGenerator &move, const C1 i, const C2 j) {

            const auto c_iv_jv = this->moves.get_edge_cost(move);

            const auto c_iprevprevprev_jprevprevprev = this->instance.get_cost(i.prevprevprev, j.prevprevprev);

            const auto delta = this->instance.get_cost(j.prevprevprevprev, i.prevprev) + c_iv_jv + c_iprevprevprev_jprevprevprev +
                               this->instance.get_cost(j.prev, i.next) + i.seqrem1 + j.seqrem2;

            return delta;
        }
    };

}  // namespace cobra

#endif