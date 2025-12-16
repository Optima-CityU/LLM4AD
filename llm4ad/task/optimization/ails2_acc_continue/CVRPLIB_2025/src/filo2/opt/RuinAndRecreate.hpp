#ifndef _FILO2_RUINANDRECREATE_HPP_
#define _FILO2_RUINANDRECREATE_HPP_

#include <algorithm>
#include <random>

#include "../base/SparseIntSet.hpp"
#include "../instance/Instance.hpp"
#include "../solution/Solution.hpp"

class RuinAndRecreate {

public:
    RuinAndRecreate(const cobra::Instance& instance_, std::mt19937& rand_engine_)
        : instance(instance_)
        , rand_engine(rand_engine_)
        , boolean_dist(std::uniform_int_distribution(0, 1))
        , customers_distribution(instance.get_customers_begin(), instance.get_customers_end() - 1)
        , rand_uniform(0, 3)
        , routes(instance.get_vertices_num()) { }

    int apply(cobra::Solution& solution, std::vector<int>& omega) {

        assert(solution.is_feasible());

        removed.clear();
        routes.clear();

        auto seed = customers_distribution(rand_engine);

        const auto N = omega[seed];

        auto curr = seed;

        for (auto n = 0; n < N; n++) {

            assert(curr != instance.get_depot());

            auto next = cobra::Solution::dummy_vertex;

            auto route = solution.get_route_index(curr);

            removed.push_back(curr);
            routes.insert(route);

            if (solution.get_route_size(route) > 1 && boolean_dist(rand_engine)) {
                // Move within the current route.

                if (boolean_dist(rand_engine)) {
                    next = solution.get_next_vertex(curr);
                    if (next == instance.get_depot()) {
                        next = solution.get_next_vertex(route, next);
                    }
                } else {
                    next = solution.get_prev_vertex(curr);
                    if (next == instance.get_depot()) {
                        next = solution.get_prev_vertex(route, next);
                    }
                }

            } else {
                // Jump to a neighbor route.

                if (boolean_dist(rand_engine)) {
                    // It could be a route that we have not yet visited.

                    for (auto m = 1u; m < instance.get_neighbors_of(curr).size(); m++) {
                        const auto neighbor = instance.get_neighbors_of(curr)[m];
                        if (neighbor == instance.get_depot() || !solution.is_customer_in_solution(neighbor) ||
                            routes.contains(solution.get_route_index(neighbor))) {
                            continue;
                        }
                        next = neighbor;
                        break;
                    }

                } else {
                    // It could be any route, even one where we have already removed some customers.

                    for (auto m = 1u; m < instance.get_neighbors_of(curr).size(); m++) {
                        const auto neighbor = instance.get_neighbors_of(curr)[m];
                        if (neighbor == instance.get_depot() || !solution.is_customer_in_solution(neighbor)) {
                            continue;
                        }
                        next = neighbor;
                        break;
                    }
                }
            }

            assert(next != instance.get_depot());

            solution.remove_vertex(route, curr);
            if (solution.is_route_empty(route)) {
                solution.remove_route(route);
            }

            if (next == cobra::Solution::dummy_vertex) {
                break;
            }

            curr = next;
        }

        // Shuffle customers around a little.
        switch (rand_uniform(rand_engine)) {
        case 0:
            std::shuffle(removed.begin(), removed.end(), rand_engine);
            break;
        case 1:
            std::sort(removed.begin(), removed.end(), [this](int a, int b) { return instance.get_demand(a) > instance.get_demand(b); });
            break;
        case 2:
            std::sort(removed.begin(), removed.end(), [this](int a, int b) {
                return instance.get_cost(a, instance.get_depot()) > instance.get_cost(b, instance.get_depot());
            });
            break;
        case 3:
            std::sort(removed.begin(), removed.end(), [this](int a, int b) {
                return instance.get_cost(a, instance.get_depot()) < instance.get_cost(b, instance.get_depot());
            });
            break;
        }

        assert(solution.is_feasible());

        // Insert removed customers back into the solution.
        for (auto customer : removed) {

            assert(customer != instance.get_depot());

            auto best_route = cobra::Solution::dummy_route;
            auto best_where = cobra::Solution::dummy_vertex;
            auto best_cost = std::numeric_limits<double>::max();

            const auto& neighbors = instance.get_neighbors_of(customer);
            routes.clear();

            // We only try to insert customers back into routes serving neighboring customers. That's not necessarily the smartest choice,
            // especially for very long routes, but seems to be working well enough.
            for (int n = 1; n < static_cast<int>(neighbors.size()); n++) {
                int where = neighbors[n];
                if (where == instance.get_depot() || !solution.is_customer_in_solution(where)) continue;
                routes.insert(solution.get_route_index(where));
            }

            const auto c_customer_depot = instance.get_cost(customer, instance.get_depot());

            for (auto route : routes.get_elements()) {

                if (solution.get_route_load(route) + instance.get_demand(customer) > instance.get_vehicle_capacity()) {
                    continue;
                }

                for (auto where = solution.get_first_customer(route); where != instance.get_depot();
                     where = solution.get_next_vertex(where)) {

                    const auto prev = solution.get_prev_vertex(where);

                    const auto cost = -solution.get_cost_prev_customer(where) + instance.get_cost(prev, customer) +
                                      instance.get_cost(customer, where);

                    if (cost < best_cost) {

                        best_cost = cost;
                        best_route = route;
                        best_where = where;
                    }
                }

                const auto cost = -solution.get_cost_prev_depot(route) + instance.get_cost(solution.get_last_customer(route), customer) +
                                  c_customer_depot;

                if (cost < best_cost) {

                    best_cost = cost;
                    best_route = route;
                    best_where = instance.get_depot();
                }
            }

            if (best_route == solution.get_end_route() || (2.0 * c_customer_depot < best_cost)) {
                solution.build_one_customer_route(customer);
            } else {
                solution.insert_vertex_before(best_route, best_where, customer);
            }

            assert(solution.is_feasible());
        }

        return seed;
    }

private:
    const cobra::Instance& instance;
    std::mt19937& rand_engine;
    std::uniform_int_distribution<int> boolean_dist;
    std::uniform_int_distribution<int> customers_distribution;
    std::uniform_int_distribution<int> rand_uniform;
    std::vector<int> removed;
    cobra::SparseIntSet routes;
};

#endif