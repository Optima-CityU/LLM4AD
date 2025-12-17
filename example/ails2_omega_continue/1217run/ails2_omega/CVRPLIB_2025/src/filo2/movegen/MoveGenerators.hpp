#ifndef _FILO2_MOVEGENERATORS_HPP_
#define _FILO2_MOVEGENERATORS_HPP_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "../base/BinaryHeap.hpp"
#include "../base/Flat2DVector.hpp"
#include "../base/SparseIntSet.hpp"
#include "../base/VectorView.hpp"
#include "../instance/Instance.hpp"

namespace cobra {

    // Simple generator of incremental numbers.
    class TimestampGenerator : private NonCopyable<TimestampGenerator> {
    public:
        TimestampGenerator() = default;
        inline unsigned long get() {
            return value;
        }
        inline void increment() {
            ++value;
        }

    private:
        unsigned long value = 0;
    };

    // Class representing a move generator or static move descriptor.
    class MoveGenerator : private NonCopyable<MoveGenerator> {
    public:
        MoveGenerator(int i, int j) : i(i), j(j) { }

        inline auto get_first_vertex() const {
            return i;
        }
        inline auto get_second_vertex() const {
            return j;
        }
        inline auto get_delta() const {
            return delta;
        }
        inline auto set_delta(double value) {
            delta = value;
        }
        inline auto get_heap_index() const {
            return heap_index;
        }
        inline auto set_heap_index(int index) {
            heap_index = index;
        }
        inline bool is_computed_for_ejch() const {
            return computed_for_ejch;
        }
        inline void set_computed_for_ejch(bool value) {
            computed_for_ejch = value;
        }

    private:
        int i;
        int j;
        double delta = 0.0;
        int heap_index = -1;
        bool computed_for_ejch = false;
    };

    struct MGCompare {
        auto operator()(MoveGenerator* mg1, MoveGenerator* mg2) {
            assert(mg1 && mg2);
            return mg1->get_delta() - mg2->get_delta();
        }
    };

    struct MGGetIdx {
        auto operator()(MoveGenerator* mg1) {
            assert(mg1);
            return mg1->get_heap_index();
        }
    };

    struct MGSetIdx {
        void operator()(MoveGenerator* mg1, int idx) {
            assert(mg1);
            mg1->set_heap_index(idx);
        }
    };

    struct MGUpdate {
        auto operator()(MoveGenerator* mg1, double delta) {
            assert(mg1);
            const auto res = mg1->get_delta() - delta;
            mg1->set_delta(delta);
            return res;
        }
    };

    // Heap data structure specialized to contain move generators.
    class MoveGeneratorsHeap
        : private NonCopyable<MoveGeneratorsHeap>
        , private BinaryHeap<MoveGenerator*, MGCompare, MGGetIdx, MGSetIdx, MGUpdate, -1> {

        typedef BinaryHeap<MoveGenerator*, MGCompare, MGGetIdx, MGSetIdx, MGUpdate> BHeap;

    public:
        MoveGeneratorsHeap() = default;
        MoveGeneratorsHeap(const MoveGeneratorsHeap& other) = delete;


        void reset() {
            BHeap::reset();
        }
        bool is_empty() const {
            return BHeap::empty();
        }
        void insert(MoveGenerator* mg) {
            BHeap::insert(mg);
        }
        MoveGenerator* get() {
            return BHeap::get();
        }
        void remove(int heap_index) {
            BHeap::remove(heap_index);
        }
        void change_value(int heap_index, double value) {
            BHeap::update(heap_index, value);
        };
        int size() const {
            return BHeap::size();
        };
        MoveGenerator* spy(int heap_index) {
            return BHeap::spy(heap_index);
        }

        static const int unheaped = -1;

    private:
        void dump() override {
            for (auto n = 0; n < size(); n++) {
                const auto& move = spy(n);
                std::cout << "[" << n << "] (" << move->get_first_vertex() << ", " << move->get_second_vertex()
                          << ") delta = " << move->get_delta() << " heap index = " << move->get_heap_index() << "\n";
            }
        }
    };

    // K-nearest neighbors move generators.
    class MoveGenerators : private NonCopyable<MoveGenerators> {
    public:
        MoveGenerators(const Instance& instance, int k)
            :  // -1 as neighbors[0] == i and we skip (i, i) move gens.
            max_num_neighbors(std::min(k, instance.get_vertices_num() - 1))
            , heap(MoveGeneratorsHeap())
            , vertex_timestamp(instance.get_vertices_num(), 0)
            , vertices_in_updated_moves(instance.get_vertices_num())
            , unique_endpoints(instance.get_vertices_num()) {

            update_bits.resize(instance.get_vertices_num(), 2);

            base_move_indices_involving.resize(instance.get_vertices_num());
            active_move_indices_involving_1st.resize(instance.get_vertices_num());
            current_num_neighbors.resize(instance.get_vertices_num(), 0);

            const int neighbors_begin = 1;
            const int neighbors_end = neighbors_begin + max_num_neighbors;

            for (int i = instance.get_vertices_begin(); i < instance.get_vertices_end(); ++i) {
                const auto& ineighbors = instance.get_neighbors_of(i);

                for (int p = neighbors_begin; p < neighbors_end; ++p) {
                    assert(p < static_cast<int>(ineighbors.size()));
                    const int j = ineighbors[p];
                    const double cost = instance.get_cost(i, j);

                    const auto insert = [this, cost](int a, int b) {
                        const int base_idx = moves.size();
                        assert(base_idx == get_base_move_generator_index(base_idx));
                        moves.emplace_back(a, b);
                        moves.emplace_back(b, a);
                        edge_costs.emplace_back(cost);
                        assert(std::fabs(get_edge_cost(moves[base_idx]) - cost) < 0.01);
                        assert(std::fabs(get_edge_cost(moves[base_idx + 1]) - cost) < 0.01);
                        base_move_indices_involving[a].emplace_back(base_idx);
                        base_move_indices_involving[b].emplace_back(base_idx);
                        assert(moves[base_idx].get_first_vertex() == a);
                        assert(moves[base_idx + 1].get_first_vertex() == b);
                    };

                    assert(i != j);
                    if (i < j) {
                        insert(i, j);
                        continue;
                    }

                    const auto& jneighbors = instance.get_neighbors_of(j);

                    const double cij = instance.get_cost(i, j);
                    const double cjn = instance.get_cost(j, jneighbors[neighbors_end - 1]);
                    if (cij > cjn) {
                        insert(j, i);
                        continue;
                    }

                    if (std::fabs(cij - cjn) < 0.00001) {
                        bool add = true;
                        for (int idx : base_move_indices_involving[j]) {
                            if (moves[idx].get_second_vertex() == i) {
                                add = false;
                                break;
                            }
                        }

                        if (add) {
                            insert(j, i);
                        }
                        continue;
                    }
                }
            }

            for (int i = instance.get_vertices_begin(); i < instance.get_vertices_end(); ++i) {
                std::sort(base_move_indices_involving[i].begin(), base_move_indices_involving[i].end(), [this](int a, int b) {
                    const auto& a_move = get(a);
                    const double a_cost = get_edge_cost(a_move);

                    const auto& b_move = get(b);
                    const double b_cost = get_edge_cost(b_move);

                    return a_cost < b_cost;
                });
            }

            // We only consider base indices (thus size / 2).
            move_active_in_1st.resize(moves.size() / 2, false);
            move_active_in_2nd.resize(moves.size() / 2, false);
        }

        inline MoveGenerator& get(int idx) {
            assert(idx >= 0 && idx < static_cast<int>(moves.size()));
            return moves[idx];
        }

        inline const MoveGenerator& get(int idx) const {
            assert(idx >= 0 && idx < static_cast<int>(moves.size()));
            return moves[idx];
        }


        inline const auto& get_move_generator_indices_involving_1st(int vertex) const {
            return active_move_indices_involving_1st[vertex];
        }

        inline auto get_move_generator_indices_involving_2nd(int vertex) const {
            const auto& v = active_move_indices_involving_1st[vertex];
            return VectorView<decltype(v.begin()), twin_functor>(v.begin(), v.end());
        }

        inline auto get_move_generator_indices_involving(int vertex) const {
            const auto& v = active_move_indices_involving_1st[vertex];
            return VectorView<decltype(v.begin()), base_functor>(v.begin(), v.end());
        }

        void set_active_percentage(std::vector<double>& percentage, std::vector<int>& vertices) {
            vertices_getting_updated.clear();
            vertices_in_updated_moves.clear();

            for (const int vertex : vertices) {
                // Convert the percentage to the number of neighbors to consider.
                const int num_neigbors = std::round(percentage[vertex] * max_num_neighbors);
                assert(num_neigbors <= static_cast<int>(base_move_indices_involving[vertex].size()));

                // Check if we are already using the requested number of neighbors.
                if (num_neigbors == current_num_neighbors[vertex]) {
                    continue;
                }

                vertices_getting_updated.push_back(vertex);

                if (num_neigbors < current_num_neighbors[vertex]) {
                    for (int n = num_neigbors; n < current_num_neighbors[vertex]; ++n) {
                        const int idx = base_move_indices_involving[vertex][n];
                        const MoveGenerator& move = moves[idx];

                        assert(is_active_in(move, vertex));
                        set_not_active_in(move, vertex);

                        vertices_in_updated_moves.insert(move.get_first_vertex());
                        vertices_in_updated_moves.insert(move.get_second_vertex());
                    }
                } else {
                    for (int n = current_num_neighbors[vertex]; n < num_neigbors; ++n) {
                        const int idx = base_move_indices_involving[vertex][n];
                        const MoveGenerator& move = moves[idx];

                        assert(!is_active_in(move, vertex));
                        set_active_in(move, vertex);

                        vertices_in_updated_moves.insert(move.get_first_vertex());
                        vertices_in_updated_moves.insert(move.get_second_vertex());
                    }
                }

                current_num_neighbors[vertex] = num_neigbors;
            }

            unique_move_generators.clear();
            unique_endpoints.clear();

            for (const int vertex : vertices_in_updated_moves.get_elements()) {
                unique_move_generators.clear();
                unique_endpoints.clear();

                // We need to scan all base move indices as some movegen may be active due to the other vertex.
                for (int base_idx : base_move_indices_involving[vertex]) {
                    assert(base_idx == get_base_move_generator_index(base_idx));
                    const auto& move = moves[base_idx];

                    if (!is_active_in_any(move)) {
                        continue;
                    }

                    const int idx = vertex == move.get_first_vertex() ? base_idx : get_twin_move_generator_index(base_idx);

                    const int j = moves[idx].get_second_vertex();
                    if (!unique_endpoints.contains(j)) {
                        unique_endpoints.insert_without_checking_existance(j);
                        unique_move_generators.push_back(idx);
                    }
                }
                active_move_indices_involving_1st[vertex] = unique_move_generators;
            }
        }

        inline MoveGeneratorsHeap& get_heap() {
            return heap;
        }

        static inline int get_twin_move_generator_index(int index) {
            return index ^ 1;
        }

        static inline int get_base_move_generator_index(int index) {
            return index & ~1;
        }

        inline TimestampGenerator& get_timestamp_generator() {
            return timegen;
        }

        inline std::vector<unsigned long>& get_vertex_timestamp() {
            return vertex_timestamp;
        }

        inline Flat2DVector<bool>& get_update_bits() {
            return update_bits;
        }

        inline double get_edge_cost(const MoveGenerator& move) {
            return edge_costs[(&move - moves.data()) / 2];
        }

        inline size_t size() {
            return moves.size();
        }

    private:
        // Sets that move is active because of vertex.
        inline void set_active_in(const MoveGenerator& move, int vertex) {
            const int idx = (&move - moves.data()) / 2;
            assert(vertex == move.get_first_vertex() || vertex == move.get_second_vertex());
            if (vertex == move.get_first_vertex()) {
                move_active_in_1st[idx] = true;
            } else {
                move_active_in_2nd[idx] = true;
            }
        }

        // Sets that move is no longer active because of vertex.
        inline void set_not_active_in(const MoveGenerator& move, int vertex) {
            const int idx = (&move - moves.data()) / 2;
            assert(vertex == move.get_first_vertex() || vertex == move.get_second_vertex());
            if (vertex == move.get_first_vertex()) {
                move_active_in_1st[idx] = false;
            } else {
                move_active_in_2nd[idx] = false;
            }
        }

        inline bool is_active_in(const MoveGenerator& move, int vertex) const {
            const int idx = (&move - moves.data()) / 2;
            assert(vertex == move.get_first_vertex() || vertex == move.get_second_vertex());
            return vertex == move.get_first_vertex() ? move_active_in_1st[idx] : move_active_in_2nd[idx];
        }

        inline bool is_active_in_any(const MoveGenerator& move) const {
            const int idx = (&move - moves.data()) / 2;
            return move_active_in_1st[idx] || move_active_in_2nd[idx];
        }

        const int max_num_neighbors;

        // Vector of move generators. In even position it contains move gen (i, j), and in odd positions move gen (j, i).
        std::vector<MoveGenerator> moves;

        // Containes all the even indexed move generator indices associated with every vertex.
        // Use get_twin_move_generator_index to get the twin move generator index.
        std::vector<std::vector<int>> base_move_indices_involving;

        // Indices of *active* move generators (i, j) for every vertex i.
        std::vector<std::vector<int>> active_move_indices_involving_1st;

        // Edge cost of the associated move generator. Since costs are symmetric and we are storing both move generators (ij) and (ji)
        // consecutively, the vector edge_cost contains c(ij)=c(ji) once only.
        std::vector<double> edge_costs;

        // For every vertex, the number of neighbors currently active
        std::vector<int> current_num_neighbors;

        // For every move generators, these vectors track whether they are active because of node i (first), node j (second), or both.
        std::vector<bool> move_active_in_1st;
        std::vector<bool> move_active_in_2nd;

        // Stores ordered move generators during local search applications.
        MoveGeneratorsHeap heap;

        // The `updated_bits` are used by local search operators to identify whether `(i, j)`, `(j, i)` or both require an update upon a
        // move execution.
        Flat2DVector<bool> update_bits;

        // Used by local search operators to identify whether move generators of a given vertex have already been updated.
        std::vector<unsigned long> vertex_timestamp;
        TimestampGenerator timegen;

        // The variables below are stored here to avoid re-allocations.
        // Vertices that are updated given the current number of neighbors and the required one (computed from the percentage vector).
        std::vector<int> vertices_getting_updated;
        SparseIntSet vertices_in_updated_moves;
        std::vector<int> unique_move_generators;
        SparseIntSet unique_endpoints;
    };


}  // namespace cobra

#endif