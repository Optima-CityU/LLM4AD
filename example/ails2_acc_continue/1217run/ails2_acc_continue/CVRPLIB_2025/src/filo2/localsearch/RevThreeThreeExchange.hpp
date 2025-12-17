#ifndef _FILO2_REVTHREETHREEEXCHANGE_HPP_
#define _FILO2_REVTHREETHREEEXCHANGE_HPP_

#include "AbstractOperator.hpp"

namespace cobra {

    template <bool reverse_both_strings = false>
    class RevThreeThreeExchange : public AbstractOperator {
    public:
        RevThreeThreeExchange(const Instance &instance_, MoveGenerators &moves_, double tolerance_)
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

            const auto jNext = solution.get_next_vertex(jRoute, j);
            const auto jNextNext = solution.get_next_vertex(jRoute, jNext);
            const auto jNextNextNext = solution.get_next_vertex(jRoute, jNextNext);
            const auto jNextNextNextNext = solution.get_next_vertex(jRoute, jNextNextNext);

            const auto iSequenceRem = -solution.get_cost_prev_vertex(iRoute, iPrevPrev) - solution.get_cost_prev_vertex(iRoute, iNext);
            const auto jSequenceRem = -solution.get_cost_prev_vertex(jRoute, jNext) -
                                      solution.get_cost_prev_vertex(jRoute, jNextNextNextNext);

            const auto iSequenceAdd = +this->instance.get_cost(jNextNextNextNext, iPrevPrev) + this->instance.get_cost(i, j);

            double jSequenceAdd;
            if constexpr (reverse_both_strings) {
                jSequenceAdd = +this->instance.get_cost(iPrevPrevPrev, jNextNextNext) + this->instance.get_cost(jNext, iNext);
            } else {
                jSequenceAdd = +this->instance.get_cost(iPrevPrevPrev, jNext) + this->instance.get_cost(jNextNextNext, iNext);
            }

            return iSequenceAdd + jSequenceAdd + iSequenceRem + jSequenceRem;
        }

        bool is_feasible(Solution &solution, const MoveGenerator &move) override {
            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iRoute = solution.get_route_index(i, j);
            const auto jRoute = solution.get_route_index(j, i);

            const auto iPrev = solution.get_prev_vertex(iRoute, i);
            const auto iPrevPrev = solution.get_prev_vertex(iRoute, iPrev);

            const auto jNext = solution.get_next_vertex(jRoute, j);
            const auto jNextNext = solution.get_next_vertex(jRoute, jNext);
            const auto jNextNextNext = solution.get_next_vertex(jRoute, jNextNext);

            return (iRoute != jRoute && iPrev != this->instance.get_depot() && iPrevPrev != this->instance.get_depot() &&
                    jNext != this->instance.get_depot() && jNextNext != this->instance.get_depot() &&
                    jNextNextNext != this->instance.get_depot() &&
                    solution.get_route_load(jRoute) - this->instance.get_demand(jNext) - this->instance.get_demand(jNextNext) -
                            this->instance.get_demand(jNextNextNext) + this->instance.get_demand(i) + this->instance.get_demand(iPrev) +
                            this->instance.get_demand(iPrevPrev) <=
                        this->instance.get_vehicle_capacity() &&
                    solution.get_route_load(iRoute) + this->instance.get_demand(jNext) + this->instance.get_demand(jNextNext) +
                            this->instance.get_demand(jNextNextNext) - this->instance.get_demand(i) - this->instance.get_demand(iPrev) -
                            this->instance.get_demand(iPrevPrev) <=
                        this->instance.get_vehicle_capacity()) ||
                   (iRoute == jRoute && j != iPrev && j != iPrevPrev && jNext != iPrevPrev && jNextNext != iPrevPrev &&
                    jNextNextNext != iPrevPrev && jNextNextNext != solution.get_prev_vertex(iRoute, iPrevPrev));
        }

        inline void execute(Solution &solution, const MoveGenerator &move, SparseIntSet &storage) override {

            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iRoute = solution.get_route_index(i, j);
            const auto jRoute = solution.get_route_index(j, i);

            const auto iPrev = solution.get_prev_vertex(iRoute, i);
            const auto iPrevPrev = solution.get_prev_vertex(iRoute, iPrev);
            const auto iPrevPrevPrev = solution.get_prev_vertex(iRoute, iPrevPrev);
            const auto iPrevPrevPrevPrev = solution.get_prev_vertex(iRoute, iPrevPrevPrev);
            const auto iPrevPrevPrevPrevPrev = solution.get_prev_vertex(iRoute, iPrevPrevPrevPrev);
            const auto iPrevPrevPrevPrevPrevPrev = solution.get_prev_vertex(iRoute, iPrevPrevPrevPrevPrev);

            const auto iNext = solution.get_next_vertex(iRoute, i);
            const auto iNextNext = solution.get_next_vertex(iRoute, iNext);
            const auto iNextNextNext = solution.get_next_vertex(iRoute, iNextNext);

            const auto jPrev = solution.get_prev_vertex(jRoute, j);
            const auto jPrevPrev = solution.get_prev_vertex(jRoute, jPrev);
            const auto jPrevPrevPrev = solution.get_prev_vertex(jRoute, jPrevPrev);


            const auto jNext = solution.get_next_vertex(jRoute, j);
            const auto jNextNext = solution.get_next_vertex(jRoute, jNext);
            const auto jNextNextNext = solution.get_next_vertex(jRoute, jNextNext);
            const auto jNextNextNextNext = solution.get_next_vertex(jRoute, jNextNextNext);
            const auto jNextNextNextNextNext = solution.get_next_vertex(jRoute, jNextNextNextNext);
            const auto jNextNextNextNextNextNext = solution.get_next_vertex(jRoute, jNextNextNextNextNext);

            storage.insert(iPrevPrevPrevPrevPrevPrev);
            storage.insert(iPrevPrevPrevPrevPrev);
            storage.insert(iPrevPrevPrevPrev);
            storage.insert(iPrevPrevPrev);
            storage.insert(iPrevPrev);
            storage.insert(iPrev);
            storage.insert(i);
            storage.insert(iNext);
            storage.insert(iNextNext);
            storage.insert(iNextNextNext);
            storage.insert(jPrevPrevPrev);
            storage.insert(jPrevPrev);
            storage.insert(jPrev);
            storage.insert(j);
            storage.insert(jNext);
            storage.insert(jNextNext);
            storage.insert(jNextNextNext);
            storage.insert(jNextNextNextNext);
            storage.insert(jNextNextNextNextNext);
            storage.insert(jNextNextNextNextNextNext);

            this->update_bits.at(iPrevPrevPrevPrevPrevPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iPrevPrevPrevPrevPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iPrevPrevPrevPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iPrevPrevPrev, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iPrevPrevPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iPrevPrev, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iPrevPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iPrev, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(i, UPDATE_BITS_FIRST, true);
            this->update_bits.at(i, UPDATE_BITS_SECOND, true);
            this->update_bits.at(iNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iNextNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(iNextNextNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jNextNextNextNextNextNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jNextNextNextNextNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jNextNextNextNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jNextNextNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jNextNextNext, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jNextNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jNextNext, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jNext, UPDATE_BITS_FIRST, true);
            this->update_bits.at(jNext, UPDATE_BITS_SECOND, true);
            this->update_bits.at(j, UPDATE_BITS_FIRST, true);
            this->update_bits.at(j, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jPrevPrev, UPDATE_BITS_SECOND, true);
            this->update_bits.at(jPrevPrevPrev, UPDATE_BITS_SECOND, true);


            solution.remove_vertex(iRoute, i);
            solution.remove_vertex(iRoute, iPrev);
            solution.remove_vertex(iRoute, iPrevPrev);

            solution.insert_vertex_before(jRoute, jNextNextNextNext, i);
            solution.insert_vertex_before(jRoute, jNextNextNextNext, iPrev);
            solution.insert_vertex_before(jRoute, jNextNextNextNext, iPrevPrev);

            solution.remove_vertex(jRoute, jNext);
            solution.remove_vertex(jRoute, jNextNext);
            solution.remove_vertex(jRoute, jNextNextNext);


            if constexpr (reverse_both_strings) {
                solution.insert_vertex_before(iRoute, iNext, jNextNextNext);
                solution.insert_vertex_before(iRoute, iNext, jNextNext);
                solution.insert_vertex_before(iRoute, iNext, jNext);
            } else {
                solution.insert_vertex_before(iRoute, iNext, jNext);
                solution.insert_vertex_before(iRoute, iNext, jNextNext);
                solution.insert_vertex_before(iRoute, iNext, jNextNextNext);
            }
        }

        void post_processing(__attribute__((unused)) Solution &solution) override { }

        struct Cache12 {
            int v, next, prevprev, prevprevprev, nextnextnext, nextnextnextnext;
            double seq1rem, seq2rem;
        };

        inline Cache12 prepare_cache12(const Solution &solution, int vertex) {

            assert(vertex != this->instance.get_depot());
            auto c = Cache12();
            c.v = vertex;
            const auto route = solution.get_route_index(c.v);
            const auto prev = solution.get_prev_vertex(c.v);
            c.next = solution.get_next_vertex(c.v);
            c.prevprev = solution.get_prev_vertex(route, prev);
            c.prevprevprev = solution.get_prev_vertex(route, c.prevprev);
            const auto nextnext = solution.get_next_vertex(route, c.next);
            c.nextnextnext = solution.get_next_vertex(route, nextnext);
            c.nextnextnextnext = solution.get_next_vertex(route, c.nextnextnext);

            const auto c_v_next = solution.get_cost_prev_vertex(route, c.next);

            // c.v = i in (i, j)
            c.seq1rem = -solution.get_cost_prev_vertex(route, c.prevprev) - c_v_next;
            // c.v = j in (i, j)
            c.seq2rem = -c_v_next - solution.get_cost_prev_vertex(route, c.nextnextnextnext);

            return c;
        }


        inline Cache12 prepare_cache12(const Solution &solution, int vertex, int backup) {

            auto c = Cache12();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            const auto prev = solution.get_last_customer(route);
            c.next = solution.get_first_customer(route);
            c.prevprev = solution.get_prev_vertex(prev);
            c.prevprevprev = solution.get_prev_vertex(route, c.prevprev);
            const auto nextnext = solution.get_next_vertex(c.next);
            c.nextnextnext = solution.get_next_vertex(route, nextnext);
            c.nextnextnextnext = solution.get_next_vertex(route, c.nextnextnext);

            const auto c_v_next = solution.get_cost_prev_customer(c.next);

            // c.v = i in (i, j)
            c.seq1rem = -solution.get_cost_prev_vertex(route, c.prevprev) - c_v_next;
            // c.v = j in (i, j)
            c.seq2rem = -c_v_next - solution.get_cost_prev_vertex(route, c.nextnextnextnext);

            return c;
        }

        inline std::pair<double, double> compute_cost_pair(const MoveGenerator &move, const struct Cache12 i, const struct Cache12 j) {

            const auto c_iv_jv = this->moves.get_edge_cost(move);

            double seq2add;
            double seq1add;
            if constexpr (reverse_both_strings) {
                const auto c_inext_jnext = this->instance.get_cost(i.next, j.next);
                seq2add = +this->instance.get_cost(i.prevprevprev, j.nextnextnext) + c_inext_jnext;
                seq1add = +this->instance.get_cost(j.prevprevprev, i.nextnextnext) + c_inext_jnext;
            } else {
                seq2add = +this->instance.get_cost(i.prevprevprev, j.next) + this->instance.get_cost(j.nextnextnext, i.next);
                seq1add = +this->instance.get_cost(j.prevprevprev, i.next) + this->instance.get_cost(i.nextnextnext, j.next);
            }

            const auto delta1 = +this->instance.get_cost(j.nextnextnextnext, i.prevprev) + c_iv_jv + seq2add + i.seq1rem + j.seq2rem;
            const auto delta2 = +this->instance.get_cost(i.nextnextnextnext, j.prevprev) + c_iv_jv + seq1add + j.seq1rem + i.seq2rem;

            return {delta1, delta2};
        }

        struct Cache1 {
            int v, next, prevprev, prevprevprev;
            double seq1rem;
        };

        inline Cache1 prepare_cache1(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache1();
            c.v = vertex;
            const auto route = solution.get_route_index(c.v);
            const auto prev = solution.get_prev_vertex(c.v);
            c.next = solution.get_next_vertex(c.v);
            c.prevprev = solution.get_prev_vertex(route, prev);
            c.prevprevprev = solution.get_prev_vertex(route, c.prevprev);
            c.seq1rem = -solution.get_cost_prev_vertex(route, c.prevprev) - solution.get_cost_prev_vertex(route, c.next);
            return c;
        }

        inline Cache1 prepare_cache1(const Solution &solution, int vertex, int backup) {

            auto c = Cache1();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            const auto prev = solution.get_last_customer(route);
            c.next = solution.get_first_customer(route);
            c.prevprev = solution.get_prev_vertex(prev);
            c.prevprevprev = solution.get_prev_vertex(route, c.prevprev);
            c.seq1rem = -solution.get_cost_prev_vertex(route, c.prevprev) - solution.get_cost_prev_customer(c.next);

            return c;
        }

        struct Cache2 {
            int v, next, nextnextnext, nextnextnextnext;
            double seq2rem;
        };

        inline Cache2 prepare_cache2(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache2();
            c.v = vertex;
            const auto route = solution.get_route_index(c.v);
            c.next = solution.get_next_vertex(c.v);
            const auto nextnext = solution.get_next_vertex(route, c.next);
            c.nextnextnext = solution.get_next_vertex(route, nextnext);
            c.nextnextnextnext = solution.get_next_vertex(route, c.nextnextnext);
            c.seq2rem = -solution.get_cost_prev_vertex(route, c.next) - solution.get_cost_prev_vertex(route, c.nextnextnextnext);
            return c;
        }

        inline Cache2 prepare_cache2(const Solution &solution, int vertex, int backup) {

            auto c = Cache2();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            c.next = solution.get_first_customer(route);
            const auto nextnext = solution.get_next_vertex(c.next);
            c.nextnextnext = solution.get_next_vertex(route, nextnext);
            c.nextnextnextnext = solution.get_next_vertex(route, c.nextnextnext);
            c.seq2rem = -solution.get_cost_prev_customer(c.next) - solution.get_cost_prev_vertex(route, c.nextnextnextnext);
            return c;
        }

        template <typename C1, typename C2>
        inline double compute_cost(const MoveGenerator &move, const C1 i, const C2 j) {

            const auto c_iv_jv = this->moves.get_edge_cost(move);

            double seq2add;
            if constexpr (reverse_both_strings) {
                const auto c_inext_jnext = this->instance.get_cost(i.next, j.next);
                seq2add = +this->instance.get_cost(i.prevprevprev, j.nextnextnext) + c_inext_jnext;
            } else {
                seq2add = +this->instance.get_cost(i.prevprevprev, j.next) + this->instance.get_cost(j.nextnextnext, i.next);
            }

            const auto delta = +this->instance.get_cost(j.nextnextnextnext, i.prevprev) + c_iv_jv + seq2add + i.seq1rem + j.seq2rem;

            return delta;
        }
    };

}  // namespace cobra

#endif