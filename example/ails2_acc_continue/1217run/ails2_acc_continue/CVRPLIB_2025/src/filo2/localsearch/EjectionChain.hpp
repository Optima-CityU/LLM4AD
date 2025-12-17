#ifndef _F4D_EJECTIONCHAIN_HPP_
#define _F4D_EJECTIONCHAIN_HPP_

#include "../base/BinaryHeap.hpp"
#include "../base/BitMatrix.hpp"
#include "../base/SmallFlatMap.hpp"
#include "AbstractOperator.hpp"

namespace cobra {

    // An implementation of AbstractOperator for the ejection chain local search operator.
    template <int max_relocation_nodes = 25>
    class EjectionChain : public AbstractOperator {

    public:
        static constexpr bool is_symmetric = false;

        EjectionChain(const Instance &instance_, MoveGenerators &moves_, double tolerance_)
            : AbstractOperator(instance_, moves_, tolerance_), forbidden_i(max_relocation_nodes), forbidden_j(max_relocation_nodes) {
            relocation_nodes.resize(max_relocation_nodes);
        }

    protected:
        inline void pre_processing(__attribute__((unused)) Solution &solution) override { }

        inline double compute_cost(const Solution &solution, const MoveGenerator &move) override {

            const auto i = move.get_first_vertex();
            const auto j = move.get_second_vertex();

            const auto iRoute = solution.get_route_index(i, j);
            const auto jRoute = solution.get_route_index(j, i);

            const auto iPrev = solution.get_prev_vertex(iRoute, i);
            const auto iNext = solution.get_next_vertex(iRoute, i);

            const auto jPrev = solution.get_prev_vertex(jRoute, j);

            auto delta = 0.0;

            if (j != iNext) {
                delta = -solution.get_cost_prev_vertex(iRoute, i) - solution.get_cost_prev_vertex(iRoute, iNext) +
                        this->instance.get_cost(iPrev, iNext) - solution.get_cost_prev_vertex(jRoute, j) +
                        this->instance.get_cost(jPrev, i) + moves.get_edge_cost(move);
            }

            return delta;
        }

        // The feasibility step for the ejection chain is more complex than that of other operators. In this context, we use the move
        // generator as the starting point for a tree of relocations.
        bool is_feasible(Solution &solution, const MoveGenerator &generating_move) override {

            // Relocate node index: number of generated tree nodes.
            short rni = 0;

            // Reset the index of the feasible relocation chain.
            feasible_rni = -1;

            // We first check whether the current `generating_move` is already a feasible relocate move. If it is the case, we just apply it
            // without further searching for ejection chains.
            // The following code is into a block to avoid hiding subsequent variables.
            {

                auto i = generating_move.get_first_vertex();
                auto j = generating_move.get_second_vertex();

                auto iRoute = solution.get_route_index(i, j);
                auto jRoute = solution.get_route_index(j, i);

                auto iPrev = solution.get_prev_vertex(iRoute, i);
                auto iNext = solution.get_next_vertex(iRoute, i);
                auto jPrev = solution.get_prev_vertex(jRoute, j);

                assert(j != iNext);

                relocation_nodes[rni].move = &generating_move;

                if (iRoute == jRoute ||
                    (solution.get_route_load(jRoute) + this->instance.get_demand(i) <= this->instance.get_vehicle_capacity())) {
                    feasible_rni = 0;
                    relocation_nodes[0].predecessor = -1;
                    forbidden_i.reset(0);
                    forbidden_j.reset(0);
                    forbidden_i.set(0, iPrev);
                    forbidden_i.set(0, i);
                    forbidden_i.set(0, iNext);
                    forbidden_i.set(0, jPrev);
                    forbidden_i.set(0, j);
                    return true;
                }

                // If `generating_move` is not feasible by itself, we start a relocation chain.
                relocation_nodes[rni].delta_sum = generating_move.get_delta();

                forbidden_i.reset(rni);
                forbidden_i.set(rni, iPrev);
                forbidden_i.set(rni, jPrev);

                forbidden_j.reset(rni);
                forbidden_j.set(rni, i);
                forbidden_j.set(rni, iNext);
                forbidden_j.set(rni, j);

                relocation_nodes[rni].modified_routes_loads.clear();
                relocation_nodes[rni].modified_routes_loads[iRoute] = solution.get_route_load(iRoute) - this->instance.get_demand(i);
                relocation_nodes[rni].modified_routes_loads[jRoute] = solution.get_route_load(jRoute) + this->instance.get_demand(i);
                relocation_nodes[rni].predecessor = -1;

                relo_heap.reset();
                relo_heap.insert(&relocation_nodes[rni]);
                rni++;
            }

            while (!relo_heap.empty()) {

                auto &curr = *relo_heap.get();
                const auto curr_index = index_of(relocation_nodes, curr);

                // Retrieve the route from which we would like to remove some vertex.
                const auto iRoute = solution.get_route_index(curr.move->get_second_vertex());

                // Retrieve the updated `iRoute` load (`iRoute` will always be in the map).
                assert(curr.modified_routes_loads.count(iRoute));
                const auto iRoute_load = curr.modified_routes_loads[iRoute];

                // Scan `iRoute` searching for customers that when removed will make the route feasible.
                for (auto i = solution.get_first_customer(iRoute); i != this->instance.get_depot(); i = solution.get_next_vertex(i)) {

                    // Check whether removing `i` is enough to restore the `iRoute` feasibility.
                    const auto iDemand = this->instance.get_demand(i);
                    if (iRoute_load - iDemand > this->instance.get_vehicle_capacity()) {
                        continue;
                    }

                    // Check whether `i` can be removed. This takes into account route segments that were already involved in some changes.
                    // For them we do not allow further edits.
                    if (forbidden_i.is_set(curr_index, i) || forbidden_j.is_set(curr_index, i)) {
                        continue;
                    }

                    const auto iPrev = solution.get_prev_vertex(iRoute, i);
                    const auto iNext = solution.get_next_vertex(iRoute, i);

                    // Since we removed the explicit cost matrix, every access to `instance.get_cost` is extremely expensive. Especially
                    // because here it is happening in the core of this loop. We are thus using all possible ways to postpone or cache
                    // accesses to the cost matrix.
                    bool iCost_computed = false;
                    double iCost = 0.0;

                    // Retrieve active move generators where `i` is the first vertex.
                    for (const auto move_index : this->moves.get_move_generator_indices_involving_1st(i)) {

                        auto &move = this->moves.get(move_index);
                        assert(move.get_first_vertex() == i);

                        const auto j = move.get_second_vertex();

                        // Make sure `j` can the target of a relocation.
                        if (j == this->instance.get_depot() || forbidden_j.is_set(curr_index, j)) {
                            continue;
                        }

                        // We want to relocate into a different route to make space in the current one.
                        const auto jRoute = solution.get_route_index(j);
                        if (jRoute == iRoute) {
                            continue;
                        }

                        // We might have already worker with `jRoute`, we thus need to retrieve the updated load.
                        auto jRoute_load = [&] {
                            if (const auto &pair = curr.modified_routes_loads.find(jRoute); pair.first != 0) {
                                return pair.second;
                            } else {
                                return solution.get_route_load(jRoute);
                            }
                        }();

                        const auto jPrev = solution.get_prev_vertex(jRoute, j);

                        // The above cycle scans active move generators `(i, j)` that might not be initialized. We can easily check that by
                        // checking whether
                        // 1. the given move generator is heaped or
                        // 2. the move generator is not in the heap but its value was previously computed and updated.
                        if (move.get_heap_index() == MoveGeneratorsHeap::unheaped && !move.is_computed_for_ejch()) {

                            if (!iCost_computed) {
                                iCost = -solution.get_cost_prev_customer(i) - solution.get_cost_prev_vertex(iRoute, iNext) +
                                        this->instance.get_cost(iPrev, iNext);
                                iCost_computed = true;
                            }

                            const auto correct_delta = iCost - solution.get_cost_prev_customer(j) + this->instance.get_cost(jPrev, i) +
                                                       this->moves.get_edge_cost(move);
                            move.set_delta(correct_delta);
                            move.set_computed_for_ejch(true);
                            computed_for_ejch.emplace_back(move_index);
                        }
                        assert(std::abs(move.get_delta() - this->compute_cost(solution, move)) < 0.01);

                        // As a pruning strategy we only consider relocations that keep the chain improving.
                        if (move.get_delta() + curr.delta_sum > -this->tolerance) {
                            continue;
                        }

                        // If we reach this point it means that move `(i, j)` restores `iRoute` feasibily (possibly violating the `jRoute`
                        // one). We thus create a relocation node.
                        relocation_nodes[rni].move = &move;
                        relocation_nodes[rni].delta_sum = curr.delta_sum + move.get_delta();

                        forbidden_i.overwrite(curr_index, rni);
                        forbidden_i.set(rni, iPrev);
                        forbidden_i.set(rni, jPrev);

                        forbidden_j.overwrite(curr_index, rni);
                        forbidden_j.set(rni, i);
                        forbidden_j.set(rni, iNext);
                        forbidden_j.set(rni, j);

                        relocation_nodes[rni].modified_routes_loads = curr.modified_routes_loads;
                        relocation_nodes[rni].modified_routes_loads[iRoute] = iRoute_load - iDemand;
                        relocation_nodes[rni].modified_routes_loads[jRoute] = jRoute_load + iDemand;

                        relocation_nodes[rni].predecessor = curr_index;
                        relo_heap.insert(&relocation_nodes[rni]);

                        // If also `jRoute` is feasible we have found a feasible chain!
                        if (jRoute_load + iDemand <= this->instance.get_vehicle_capacity()) {
                            feasible_rni = rni;
                            goto end;
                        }

                        rni++;

                        if (rni == max_relocation_nodes) {
                            goto end;
                        }
                    }
                }
            }

        end:
            return feasible_rni != -1;
        }

        inline void execute(Solution &solution, __attribute__((unused)) const MoveGenerator &generating_move,
                            SparseIntSet &affected_vertices) override {

            for (auto i : forbidden_i.get_set_entries_possibly_with_duplicates(feasible_rni)) {
                affected_vertices.insert(i);
            }
            for (auto j : forbidden_j.get_set_entries_possibly_with_duplicates(feasible_rni)) {
                affected_vertices.insert(j);
            }

            // Reset some cached deltas associated with affected vertices.
            for (const int i : affected_vertices.get_elements()) {
                for (const int move_index : moves.get_move_generator_indices_involving(i)) {
                    moves.get(move_index).set_computed_for_ejch(false);
                    moves.get(move_index + 1).set_computed_for_ejch(false);
                }
            }

            for (auto ptr = feasible_rni; ptr != -1; ptr = relocation_nodes[ptr].predecessor) {
                const auto &move = relocation_nodes[ptr].move;

                const auto i = move->get_first_vertex();
                const auto j = move->get_second_vertex();

                const auto iRoute = solution.get_route_index(i, j);
                const auto jRoute = solution.get_route_index(j, i);

                this->update_bits.at(solution.get_prev_vertex(iRoute, i), UPDATE_BITS_FIRST, true);
                this->update_bits.at(i, UPDATE_BITS_FIRST, true);
                this->update_bits.at(i, UPDATE_BITS_SECOND, true);
                const auto iNext = solution.get_next_vertex(iRoute, i);
                this->update_bits.at(iNext, UPDATE_BITS_FIRST, true);
                this->update_bits.at(iNext, UPDATE_BITS_SECOND, true);
                this->update_bits.at(j, UPDATE_BITS_FIRST, true);
                this->update_bits.at(j, UPDATE_BITS_SECOND, true);
                this->update_bits.at(solution.get_prev_vertex(jRoute, j), UPDATE_BITS_FIRST, true);

                solution.remove_vertex(iRoute, i);
                solution.insert_vertex_before(jRoute, j, i);

                if (solution.is_route_empty(iRoute)) {
                    solution.remove_route(iRoute);
                }
            }

            assert(solution.is_feasible());
        }

        void post_processing(__attribute__((unused)) Solution &solution) override {
            // Reset remaining cached deltas.
            for (const int move_index : computed_for_ejch) {
                const int base_index = move_index & (~1);
                const int twin_index = base_index + 1;
                moves.get(base_index).set_computed_for_ejch(false);
                moves.get(twin_index).set_computed_for_ejch(false);
            }
            computed_for_ejch.clear();
        }

        struct Cache12 {
            int v, prev, next;
            double vrem, prevrem;
        };

        inline Cache12 prepare_cache12(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache12();
            c.v = vertex;
            c.prev = solution.get_prev_vertex(c.v);
            c.next = solution.get_next_vertex(c.v);
            const auto route = solution.get_route_index(c.v);

            // c.v = i in (i, j)
            c.vrem = -solution.get_cost_prev_customer(c.v) - solution.get_cost_prev_vertex(route, c.next) +
                     this->instance.get_cost(c.prev, c.next);
            // c.v = j in (i, j)
            c.prevrem = -solution.get_cost_prev_customer(c.v);

            return c;
        }

        inline Cache12 prepare_cache12(const Solution &solution, int vertex, int backup) {
            assert(backup != instance.get_depot());

            auto c = Cache12();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            c.prev = solution.get_last_customer(route);
            c.next = solution.get_first_customer(route);

            // c.v = i in (i, j)
            c.vrem = -solution.get_cost_prev_depot(route) - solution.get_cost_prev_customer(c.next) +
                     this->instance.get_cost(c.prev, c.next);
            // c.v = j in (i, j)
            c.prevrem = -solution.get_cost_prev_depot(route);

            return c;
        }

        inline std::pair<double, double> compute_cost_pair(const MoveGenerator &move, const struct Cache12 i, const struct Cache12 j) {

            const auto c_iv_jv = this->moves.get_edge_cost(move);
            const auto delta1 = j.v != i.next ? i.vrem + j.prevrem + this->instance.get_cost(j.prev, i.v) + c_iv_jv : 0.0;
            const auto delta2 = i.v != j.next ? j.vrem + i.prevrem + this->instance.get_cost(i.prev, j.v) + c_iv_jv : 0.0;

            return {delta1, delta2};
        }

        struct Cache1 {
            int v, prev, next;
            double vrem;
        };

        inline Cache1 prepare_cache1(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache1();
            c.v = vertex;
            c.prev = solution.get_prev_vertex(c.v);
            c.next = solution.get_next_vertex(c.v);
            const auto route = solution.get_route_index(c.v);

            c.vrem = -solution.get_cost_prev_customer(c.v) - solution.get_cost_prev_vertex(route, c.next) +
                     this->instance.get_cost(c.prev, c.next);

            return c;
        }

        inline Cache1 prepare_cache1(const Solution &solution, int vertex, int backup) {
            assert(backup != instance.get_depot());
            auto c = Cache1();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            c.prev = solution.get_last_customer(route);
            c.next = solution.get_first_customer(route);
            c.vrem = -solution.get_cost_prev_depot(route) - solution.get_cost_prev_customer(c.next) +
                     this->instance.get_cost(c.prev, c.next);
            return c;
        }

        struct Cache2 {
            int v, prev;
            double prevrem;
        };

        inline Cache2 prepare_cache2(const Solution &solution, int vertex) {
            assert(vertex != this->instance.get_depot());
            auto c = Cache2();
            c.v = vertex;
            c.prev = solution.get_prev_vertex(c.v);
            c.prevrem = -solution.get_cost_prev_customer(c.v);
            return c;
        }

        inline Cache2 prepare_cache2(const Solution &solution, int vertex, int backup) {
            assert(backup != instance.get_depot());
            auto c = Cache2();
            c.v = vertex;
            const auto route = solution.get_route_index(backup);
            c.prev = solution.get_last_customer(route);
            c.prevrem = -solution.get_cost_prev_depot(route);
            return c;
        }

        template <typename C1, typename C2>
        inline double compute_cost(const MoveGenerator &move, const C1 i, const C2 j) {
            return j.v != i.next ? i.vrem + j.prevrem + this->instance.get_cost(j.prev, i.v) + this->moves.get_edge_cost(move) : 0.0;
        }

    private:
        static constexpr int heap_unheaped = -1;
        static constexpr auto max_chain_length = max_relocation_nodes;

        // Relocation node.
        struct Relocation {
            short heap_index = heap_unheaped;
            short predecessor = 0;
            double delta_sum = 0.0;
            const MoveGenerator *move = nullptr;
            SmallFlatMap<int, int, 0, 25> modified_routes_loads;
        };

        BitMatrix<2 * max_chain_length + 3> forbidden_i;
        BitMatrix<3 * max_chain_length> forbidden_j;

        std::vector<Relocation> relocation_nodes;
        short feasible_rni;

        std::vector<int> computed_for_ejch;

        BinaryHeapPtr<Relocation, &Relocation::heap_index, &Relocation::delta_sum> relo_heap;
        inline int index_of(std::vector<Relocation> &nodes, Relocation &node) {
            return &node - nodes.data();
        }
    };

}  // namespace cobra

#endif
