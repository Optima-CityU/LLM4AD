#ifndef _FILO2_ABSTRACTOPERATOR_HPP_
#define _FILO2_ABSTRACTOPERATOR_HPP_

#include "../instance/Instance.hpp"
#include "../movegen/MoveGenerators.hpp"
#include "../solution/Solution.hpp"

// Update bits are used to restrict the update of SMDs for asymmetric neighborhoods. After a local search move is executed, if
// `updated_bits.at(i, UPDATE_BITS_FIRST)` is true for an affected vertex `i` identifies that move generators `(i, j)` where `i` is in first
// position need to be updated. Similarly, `updated_bits.at(i, UPDATE_BITS_SECOND)` identifies that move generators `(j, i)` where `i` is
// the second vertex require an update.
//
// For more info refer to Appendix B.2.0.1 of the FILO paper.
#define UPDATE_BITS_FIRST (0)
#define UPDATE_BITS_SECOND (1)

namespace cobra {

    // An abstract local search operator based on SMDs and GNs.
    class AbstractOperator : public NonCopyable<AbstractOperator> {
    public:
        AbstractOperator(const Instance& instance_, MoveGenerators& moves_, double tolerance_)
            : instance(instance_), moves(moves_), heap(moves.get_heap()), tolerance(tolerance_), update_bits(moves_.get_update_bits()) { }
        virtual ~AbstractOperator() = default;

        // Applies the operator to `solution`.
        virtual bool apply_rough_best_improvement(Solution& solution) = 0;

    protected:
        // Performs some setup useful during a local search cycle, if necessary.
        virtual void pre_processing(Solution& solution) = 0;

        // Computes the effect of `move` to `solution` without caring about whether its application would make `solution` infeasible. This
        // function is currently used only for debug purposes. For the actual move cost computation see for example `symmetric_init`.
        virtual double compute_cost(const Solution& solution, const MoveGenerator& move) = 0;

        // Returns whether applying `move` to `solution` would create a feasible solution.
        virtual bool is_feasible(Solution& solution, const MoveGenerator& move) = 0;

        // Applies `move` to `solution` and populates `affected_vertices` with the vertices `i` for which at least one move generator `(i,
        // j)` or `(j, i)` require an update.
        virtual void execute(Solution& solution, const MoveGenerator& move, SparseIntSet& affected_vertices) = 0;

        // Performs some final cleanup at the end of the local search cycle, if necessary.
        virtual void post_processing(Solution& solution) = 0;

        const Instance& instance;
        MoveGenerators& moves;
        MoveGeneratorsHeap& heap;
        const double tolerance;

        // It allows to identify whether `(i, j)`, `(j, i)` or both require an update after a move execution. Values are set during
        // `execute` for vertices affected by the move application. This data structure is shared across operator to minimize memory
        // occupation.
        Flat2DVector<bool>& update_bits;
    };

    // A base local search operator containing code which is operator independent such as SMDs initialization, update and search logic.
    // Templating is used to perform compile time inheritance.
    template <class T, bool handle_partial_solutions = false>
    class CommonOperator : public T {

    public:
        CommonOperator(const Instance& instance_, MoveGenerators& moves_, double tolerance_)
            : T(instance_, moves_, tolerance_)
            , timegen(moves_.get_timestamp_generator())
            , affected_vertices(instance_.get_vertices_num()) { }

        // Applies a local search cycle to `solution` and returns whether there were some improvements.
        bool apply_rough_best_improvement(Solution& solution) {
            T::heap.reset();

            T::pre_processing(solution);

            initialize_descriptors(solution);
            assert(__check_deltas(solution));


            auto improved = false;

            auto index = 0;

            while (index < T::heap.size()) {
                auto& move = *T::heap.spy(index++);

                if constexpr (handle_partial_solutions) {
                    if (!solution.is_vertex_in_solution(move.get_first_vertex()) ||
                        !solution.is_vertex_in_solution(move.get_second_vertex())) {
                        continue;
                    }
                }

                if (!T::is_feasible(solution, move)) {
                    continue;
                }

#ifndef NDEBUG
                auto old_cost = solution.get_cost();
#endif

                T::execute(solution, move, affected_vertices);

                assert(old_cost > solution.get_cost());
                assert(solution.is_feasible());

                improved = true;
                index = 0;

                descriptors_update(solution);
                assert(__check_deltas(solution));

                affected_vertices.clear();
            }

            T::post_processing(solution);

            return improved;
        }


    private:
        // Check the delta of heaped moves.
        bool __check_deltas(Solution& solution) {
            for (int i = 0; i < T::heap.size(); ++i) {
                auto& move = *T::heap.spy(i);

                if (!T::is_feasible(solution, move)) {
                    // Skip non feasible moves since the delta is wrong for inter-route only moves such as split and tails when i and j
                    // belong to the same route.
                    continue;
                }
                if (std::fabs(move.get_delta() - T::compute_cost(solution, move)) > 0.01) {
                    std::cout << "Operator: " << typeid(this).name() << "\n";
                    std::cout << "Error: delta cost mismatch!!\n";
                    std::cout << "Stored = " << move.get_delta() << "\n";
                    std::cout << "Computed = " << T::compute_cost(solution, move) << "\n";
                    int i = move.get_first_vertex();
                    int j = move.get_second_vertex();
                    std::cout << "\ti=" << i << "\n";
                    std::cout << "\tj=" << j << "\n";
                    int i_route = solution.get_route_index(i, j);
                    int j_route = solution.get_route_index(j, i);
                    solution.print(i_route);
                    solution.print(j_route);
                    return false;
                }
            }
            return true;
        }

        // Initializes SMDs at the beginning of a new local search cycle.
        inline void initialize_descriptors(const Solution& solution) {
            if constexpr (T::is_symmetric) {
                symmetric_init(solution);
            } else {
                asymmetric_init(solution);
            }
        }

        // Performs an initialization for a symmetric operator. Thus only one between (i, j) and (j, i) will be initialized and used during
        // the search.
        inline void symmetric_init(const Solution& solution) {
            const auto currenttime = timegen.get() + 1;
            auto& vtimestamp = T::moves.get_vertex_timestamp();

            auto depot = false;

            // Consider only moves involving vertices in the SVC.
            for (auto i = solution.get_svc_begin(); i != solution.get_svc_end(); i = solution.get_svc_next(i)) {

                if constexpr (handle_partial_solutions) {
                    if (!solution.is_vertex_in_solution(i)) {
                        continue;
                    }
                }

                // Postpone depot handling. This way we have better chances to compact move generators and reuse cached data.
                if (i == T::instance.get_depot()) {
                    depot = true;
                    continue;
                }

                // Compute a partial move application.
                const auto icache = T::prepare_cache12(solution, i);

                // Iterate over move generators `(i, j)` so as to exploit `icache`.
                for (auto move_i1st_index : T::moves.get_move_generator_indices_involving_1st(i)) {

                    const auto& move_i1st = T::moves.get(move_i1st_index);
                    const auto j = move_i1st.get_second_vertex();

                    if constexpr (handle_partial_solutions) {
                        if (!solution.is_vertex_in_solution(j)) {
                            continue;
                        }
                    }

                    // Skip the move if we have already processed move `(j, i)`. This could happen since we might have both `i` and `j` in
                    // the SVC.
                    if (vtimestamp[j] == currenttime) {
                        continue;
                    }

                    // Always consider a fixed move generator between `(i, j)` and `(j, i)` identified by the base index.
                    const auto move_index = MoveGenerators::get_base_move_generator_index(move_i1st_index);
                    auto& move = T::moves.get(move_index);

                    // Compute the remaining partial move application.
                    const auto jcache = j == T::instance.get_depot() ? T::prepare_cache12(solution, j, i) : T::prepare_cache12(solution, j);

                    const auto delta = T::compute_cost(move, icache, jcache);

                    move.set_delta(delta);
                    move.set_heap_index(MoveGeneratorsHeap::unheaped);
                    if (move.get_delta() < -T::tolerance) {
                        T::heap.insert(&move);
                    }
                }

                vtimestamp[i] = currenttime;
            }

            // Do the same as above for the depot. Not super dry true.
            if (depot) {

                const auto i = T::instance.get_depot();

                for (auto move_i1st_index : T::moves.get_move_generator_indices_involving_1st(i)) {

                    const auto& move_i1st = T::moves.get(move_i1st_index);
                    const auto j = move_i1st.get_second_vertex();

                    if constexpr (handle_partial_solutions) {
                        if (!solution.is_vertex_in_solution(j)) {
                            continue;
                        }
                    }

                    if (vtimestamp[j] == currenttime) {
                        continue;
                    }

                    const auto move_index = MoveGenerators::get_base_move_generator_index(move_i1st_index);
                    auto& move = T::moves.get(move_index);

                    const auto icache = T::prepare_cache12(solution, i, j);
                    // Note that since `i` is the depot, `j` cannot be the depot.
                    const auto jcache = T::prepare_cache12(solution, j);

                    const auto delta = T::compute_cost(move, icache, jcache);
                    move.set_delta(delta);
                    move.set_heap_index(MoveGeneratorsHeap::unheaped);
                    if (move.get_delta() < -T::tolerance) {
                        T::heap.insert(&move);
                    }
                }

                vtimestamp[i] = currenttime;
            }

            timegen.increment();
        }

        // Performs an initialization for an asymmetric operator. Both (i, j) and (j, i) are initialized and used throghout the search.
        inline void asymmetric_init(const Solution& solution) {

            const auto currenttime = timegen.get() + 1;

            auto& vtimestamp = T::moves.get_vertex_timestamp();

            auto depot = false;

            // To make the local search localized, we only consider move generators for which at least one between `i` and `j` are cached.
            for (auto i = solution.get_svc_begin(); i != solution.get_svc_end(); i = solution.get_svc_next(i)) {

                if constexpr (handle_partial_solutions) {
                    if (!solution.is_vertex_in_solution(i)) {
                        continue;
                    }
                }

                // Postpone depot handling. This way we have better chances to compact move generators and reuse cached data.
                if (i == T::instance.get_depot()) {
                    depot = true;
                    continue;
                }

                // Compute a partial move application where `i` is either the first vertex or the second vertex.
                const auto icache = T::prepare_cache12(solution, i);

                for (const auto move_index : T::moves.get_move_generator_indices_involving_1st(i)) {

                    auto& move = T::moves.get(move_index);

                    const auto j = move.get_second_vertex();

                    if constexpr (handle_partial_solutions) {
                        if (!solution.is_vertex_in_solution(j)) {
                            continue;
                        }
                    }

                    // Skip the move if we have already processed move `(j, i)`. This could happen since we might have both `i` and `j` in
                    // the SVC.
                    if (vtimestamp[j] == currenttime) {
                        continue;
                    }

                    // Compute a partial move application where `j` is either the first or the second vertex.
                    const auto jcache = j == T::instance.get_depot() ? T::prepare_cache12(solution, j, i) : T::prepare_cache12(solution, j);

                    // Compute the actual move generator delta for (i, j) and (j, i).
                    const auto [delta1, delta2] = T::compute_cost_pair(move, icache, jcache);

                    move.set_delta(delta1);
                    move.set_heap_index(MoveGeneratorsHeap::unheaped);
                    if (move.get_delta() < -T::tolerance) {
                        T::heap.insert(&move);
                    }

                    const auto twin_move_index = MoveGenerators::get_twin_move_generator_index(move_index);
                    auto& twin_move = T::moves.get(twin_move_index);
                    twin_move.set_delta(delta2);
                    twin_move.set_heap_index(MoveGeneratorsHeap::unheaped);
                    if (twin_move.get_delta() < -T::tolerance) {
                        T::heap.insert(&twin_move);
                    }
                }

                vtimestamp[i] = currenttime;
            }

            if (depot) {

                const auto i = T::instance.get_depot();

                for (const auto move_index : T::moves.get_move_generator_indices_involving_1st(i)) {

                    auto& move = T::moves.get(move_index);

                    const auto j = move.get_second_vertex();

                    if constexpr (handle_partial_solutions) {
                        if (!solution.is_vertex_in_solution(j)) {
                            continue;
                        }
                    }

                    if (vtimestamp[j] == currenttime) {
                        continue;
                    }

                    const auto icache = T::prepare_cache12(solution, i, j);
                    // Vertex `j` cannot be the depot since `i` is the depot.
                    const auto jcache = T::prepare_cache12(solution, j);

                    const auto [delta1, delta2] = T::compute_cost_pair(move, icache, jcache);

                    move.set_delta(delta1);
                    move.set_heap_index(MoveGeneratorsHeap::unheaped);
                    if (move.get_delta() < -T::tolerance) {
                        T::heap.insert(&move);
                    }

                    const auto twin_move_index = MoveGenerators::get_twin_move_generator_index(move_index);
                    auto& twin_move = T::moves.get(twin_move_index);
                    twin_move.set_delta(delta2);
                    twin_move.set_heap_index(MoveGeneratorsHeap::unheaped);
                    if (twin_move.get_delta() < -T::tolerance) {
                        T::heap.insert(&twin_move);
                    }
                }

                vtimestamp[i] = currenttime;
            }

            timegen.increment();
        }

        // Updates affected SMDs once a move is applied.
        inline void descriptors_update(const Solution& solution) {

            if constexpr (T::is_symmetric) {
                symmetric_update(solution);
            } else {
                asymmetric_update(solution);
            }
        }

        // Performs the update for a symmetric operator.
        inline void symmetric_update(const Solution& solution) {

            const auto currenttime = timegen.get() + 1;

            auto& vtimestamp = T::moves.get_vertex_timestamp();

            const auto heap_insert = [this](MoveGenerator& move, double delta) {
                if (delta > -T::tolerance) {

                    if (move.get_heap_index() != MoveGeneratorsHeap::unheaped) {
                        T::heap.remove(move.get_heap_index());
                    }

                    move.set_delta(delta);
                } else {

                    if (move.get_heap_index() == MoveGeneratorsHeap::unheaped) {
                        move.set_delta(delta);
                        T::heap.insert(&move);
                    } else {
                        T::heap.change_value(move.get_heap_index(), delta);
                    }
                }
            };

            auto depot = false;
            for (auto i : affected_vertices.get_elements()) {

                if constexpr (handle_partial_solutions) {
                    if (!solution.is_vertex_in_solution(i)) {
                        continue;
                    }
                }

                // Postpone depot handling. This way we have better chances to compact move generators and reuse cached data.
                if (i == T::instance.get_depot()) {
                    depot = true;
                    continue;
                }

                // Compute a partial move application.
                auto icache = T::prepare_cache12(solution, i);

                // Iterate over move generators `(i, j)` so as to exploit `icache`.
                for (auto move_i1st_index : T::moves.get_move_generator_indices_involving_1st(i)) {

                    const auto& move_i1st = T::moves.get(move_i1st_index);
                    const auto j = move_i1st.get_second_vertex();

                    if constexpr (handle_partial_solutions) {
                        if (!solution.is_vertex_in_solution(j)) {
                            continue;
                        }
                    }

                    // Skip the move if we have already processed move `(j, i)`. This could happen since we might have both `i` and `j` in
                    // the SVC.
                    if (vtimestamp[j] == currenttime) {
                        continue;
                    }

                    // Always consider a fixed move generator between `(i, j)` and `(j, i)` identified by the base index.
                    const auto move_index = MoveGenerators::get_base_move_generator_index(move_i1st_index);
                    auto& move = T::moves.get(move_index);

                    const auto jcache = j == T::instance.get_depot() ? T::prepare_cache12(solution, j, i) : T::prepare_cache12(solution, j);

                    const auto delta = T::compute_cost(move, icache, jcache);
                    heap_insert(move, delta);
                }
            }

            if (depot) {

                const auto i = T::instance.get_depot();

                for (auto move_i1st_index : T::moves.get_move_generator_indices_involving_1st(i)) {

                    const auto& move_i1st = T::moves.get(move_i1st_index);
                    const auto j = move_i1st.get_second_vertex();

                    if constexpr (handle_partial_solutions) {
                        if (!solution.is_vertex_in_solution(j)) {
                            continue;
                        }
                    }

                    if (vtimestamp[j] == currenttime) {
                        continue;
                    }

                    const auto move_index = MoveGenerators::get_base_move_generator_index(move_i1st_index);
                    auto& move = T::moves.get(move_index);

                    const auto icache = T::prepare_cache12(solution, i, j);
                    // Note that `j` cannot be the depot.
                    const auto jcache = T::prepare_cache12(solution, j);

                    const auto delta = T::compute_cost(move, icache, jcache);
                    heap_insert(move, delta);
                }

                vtimestamp[i] = currenttime;
            }

            timegen.increment();
        }


        inline void asymmetric_update(const Solution& solution) {

            const auto currenttime = timegen.get() + 1;

            auto& vtimestamp = T::moves.get_vertex_timestamp();

            const auto heap_insert = [this](MoveGenerator& move, double delta) {
                if (delta > -T::tolerance) {

                    if (move.get_heap_index() != MoveGeneratorsHeap::unheaped) {
                        T::heap.remove(move.get_heap_index());
                    }

                    move.set_delta(delta);
                } else {

                    if (move.get_heap_index() == MoveGeneratorsHeap::unheaped) {
                        move.set_delta(delta);
                        T::heap.insert(&move);
                    } else {
                        T::heap.change_value(move.get_heap_index(), delta);
                    }
                }
            };

            auto depot = false;
            for (auto i : affected_vertices.get_elements()) {

                if constexpr (handle_partial_solutions) {
                    if (!solution.is_vertex_in_solution(i)) {
                        continue;
                    }
                }

                // Postpone depot handling. This way we have better chances to compact move generators and reuse cached data.
                if (i == T::instance.get_depot()) {
                    depot = true;
                    continue;
                }

                const auto iupij = T::update_bits.at(i, UPDATE_BITS_FIRST);
                const auto iupji = T::update_bits.at(i, UPDATE_BITS_SECOND);

                // Perform a restricted update guided by the update bits state.
                if (iupij && iupji) {  // Update both `(i, j)` and `(j, i)`.

                    const auto icache = T::prepare_cache12(solution, i);

                    for (const auto move_index : T::moves.get_move_generator_indices_involving_1st(i)) {

                        auto& move = T::moves.get(move_index);
                        const auto j = move.get_second_vertex();

                        if constexpr (handle_partial_solutions) {
                            if (!solution.is_vertex_in_solution(j)) {
                                continue;
                            }
                        }

                        // Move generators `(j, i)` and `(i, j)` may have been already updated. We need to check since update bits are not
                        // symmetric.
                        if (vtimestamp[j] == currenttime) {

                            const auto jupji = T::update_bits.at(j, UPDATE_BITS_FIRST);
                            const auto jupij = T::update_bits.at(j, UPDATE_BITS_SECOND);

                            if (jupji && jupij) {
                                // Both `(i, j)` and `(j, i)` were already updated.
                            } else if (jupji) {  // Move generator `(j, i)` was already updated, thus update `(i, j)` only.

                                const auto jcache = j == T::instance.get_depot() ? T::prepare_cache2(solution, j, i)
                                                                                 : T::prepare_cache2(solution, j);
                                const auto delta = T::compute_cost(move, icache, jcache);

                                heap_insert(move, delta);

                            } else if (jupij) {  // Move generator `(i, j)` was already updated, thus update `(j, i)` only.

                                const auto jcache = j == T::instance.get_depot() ? T::prepare_cache1(solution, j, i)
                                                                                 : T::prepare_cache1(solution, j);

                                const auto twin_move_index = MoveGenerators::get_twin_move_generator_index(move_index);
                                auto& twin_move = T::moves.get(twin_move_index);

                                const auto twin_delta = T::compute_cost(twin_move, jcache, icache);

                                heap_insert(twin_move, twin_delta);
                            }

                        } else {  // Move generators involving `j` were not updated before. Update both `(i, j)` and `(j, i)`.

                            const auto jcache = j == T::instance.get_depot() ? T::prepare_cache12(solution, j, i)
                                                                             : T::prepare_cache12(solution, j);

                            const auto twin_move_index = MoveGenerators::get_twin_move_generator_index(move_index);
                            auto& twin_move = T::moves.get(twin_move_index);

                            const auto [delta1, delta2] = T::compute_cost_pair(twin_move, icache, jcache);

                            heap_insert(move, delta1);
                            heap_insert(twin_move, delta2);
                        }
                    }

                } else if (iupij) {  // Update only `(i, j)`.

                    const auto icache = T::prepare_cache1(solution, i);

                    for (const auto move_index : T::moves.get_move_generator_indices_involving_1st(i)) {

                        auto& move = T::moves.get(move_index);
                        const auto j = move.get_second_vertex();

                        if constexpr (handle_partial_solutions) {
                            if (!solution.is_vertex_in_solution(j)) {
                                continue;
                            }
                        }

                        if (vtimestamp[j] != currenttime ||               // Move gen of vertex `j` not process before ...
                            (vtimestamp[j] == currenttime &&              // ... or ...
                             !T::update_bits.at(j, UPDATE_BITS_SECOND)))  // move gen of vertex `j` were processed but `(i, j)` was not
                                                                          // updated because not required.
                        {

                            const auto jcache = j == T::instance.get_depot() ? T::prepare_cache2(solution, j, i)
                                                                             : T::prepare_cache2(solution, j);
                            const auto delta = T::compute_cost(move, icache, jcache);

                            heap_insert(move, delta);
                        }
                    }

                } else if (iupji) {  // Update only `(j, i)`.

                    const auto icache = T::prepare_cache2(solution, i);

                    for (const auto move_index : T::moves.get_move_generator_indices_involving_2nd(i)) {

                        auto& move = T::moves.get(move_index);
                        const auto j = move.get_first_vertex();

                        if constexpr (handle_partial_solutions) {
                            if (!solution.is_vertex_in_solution(j)) {
                                continue;
                            }
                        }

                        if (vtimestamp[j] != currenttime ||              // Move gen of vertex `j` not process before ...
                            (vtimestamp[j] == currenttime &&             // ... or ...
                             !T::update_bits.at(j, UPDATE_BITS_FIRST)))  // move gen of vertex `j` were processed but `(j, i)` was not
                                                                         // updated because not required.
                        {

                            const auto jcache = j == T::instance.get_depot() ? T::prepare_cache1(solution, j, i)
                                                                             : T::prepare_cache1(solution, j);
                            const auto delta = T::compute_cost(move, jcache, icache);

                            heap_insert(move, delta);
                        }
                    }
                }

                vtimestamp[i] = currenttime;
            }

            // Same as above but for the depot.
            if (depot) {

                const auto i = T::instance.get_depot();

                const auto iupij = T::update_bits.at(i, UPDATE_BITS_FIRST);
                const auto iupji = T::update_bits.at(i, UPDATE_BITS_SECOND);

                if (iupij && iupji) {

                    for (const auto move_index : T::moves.get_move_generator_indices_involving_1st(i)) {

                        auto& move = T::moves.get(move_index);

                        const auto j = move.get_second_vertex();

                        if constexpr (handle_partial_solutions) {
                            if (!solution.is_vertex_in_solution(j)) {
                                continue;
                            }
                        }

                        if (vtimestamp[j] == currenttime) {

                            const auto jupji = T::update_bits.at(j, UPDATE_BITS_FIRST);
                            const auto jupij = T::update_bits.at(j, UPDATE_BITS_SECOND);

                            if (jupji && jupij) {
                            } else if (jupji) {

                                const auto icache = T::prepare_cache1(solution, i, j);
                                const auto jcache = T::prepare_cache2(solution, j);
                                const auto delta = T::compute_cost(move, icache, jcache);

                                heap_insert(move, delta);

                            } else if (jupij) {

                                const auto twin_move_index = MoveGenerators::get_twin_move_generator_index(move_index);
                                auto& twin_move = T::moves.get(twin_move_index);

                                const auto icache = T::prepare_cache2(solution, i, j);
                                const auto jcache = T::prepare_cache1(solution, j);
                                const auto twin_delta = T::compute_cost(twin_move, jcache, icache);

                                heap_insert(twin_move, twin_delta);
                            }

                        } else {

                            const auto icache = T::prepare_cache12(solution, i, j);
                            const auto jcache = T::prepare_cache12(solution, j);
                            const auto [delta1, delta2] = T::compute_cost_pair(move, icache, jcache);

                            heap_insert(move, delta1);

                            const auto twin_move_index = MoveGenerators::get_twin_move_generator_index(move_index);
                            auto& twin_move = T::moves.get(twin_move_index);
                            heap_insert(twin_move, delta2);
                        }
                    }

                } else if (iupij) {


                    for (const auto move_index : T::moves.get_move_generator_indices_involving_1st(i)) {

                        auto& move = T::moves.get(move_index);
                        const auto j = move.get_second_vertex();

                        if constexpr (handle_partial_solutions) {
                            if (!solution.is_vertex_in_solution(j)) {
                                continue;
                            }
                        }

                        if (vtimestamp[j] != currenttime || (vtimestamp[j] == currenttime && !T::update_bits.at(j, UPDATE_BITS_SECOND))) {

                            const auto icache = T::prepare_cache1(solution, i, j);
                            const auto jcache = T::prepare_cache2(solution, j);
                            const auto delta = T::compute_cost(move, icache, jcache);
                            heap_insert(move, delta);
                        }
                    }

                } else if (iupji) {

                    for (const auto move_index : T::moves.get_move_generator_indices_involving_2nd(i)) {

                        auto& move = T::moves.get(move_index);
                        const auto j = move.get_first_vertex();

                        if constexpr (handle_partial_solutions) {
                            if (!solution.is_vertex_in_solution(j)) {
                                continue;
                            }
                        }

                        if (vtimestamp[j] != currenttime || (vtimestamp[j] == currenttime && !T::update_bits.at(j, UPDATE_BITS_FIRST))) {

                            const auto icache = T::prepare_cache2(solution, i, j);
                            const auto jcache = T::prepare_cache1(solution, j);
                            const auto delta = T::compute_cost(move, jcache, icache);
                            heap_insert(move, delta);
                        }
                    }
                }

                vtimestamp[i] = currenttime;
            }

            // Reset the update bits associated with affected vertices.
            for (auto i : affected_vertices.get_elements()) {
                T::update_bits.at(i, UPDATE_BITS_FIRST, false);
                T::update_bits.at(i, UPDATE_BITS_SECOND, false);
            }

            timegen.increment();
        }

        // Generate unique increasing numbers to tag updates.
        TimestampGenerator& timegen;

        // Set of vertices affected by a move application. This set together with `update_bits` define the move generators requiring an
        // update.
        SparseIntSet affected_vertices;
    };


}  // namespace cobra
#endif
