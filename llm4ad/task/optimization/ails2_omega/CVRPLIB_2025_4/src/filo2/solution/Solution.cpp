#include "Solution.hpp"

#include <algorithm>
#include <iostream>
#include <set>

namespace cobra {

    bool Solution::is_feasible(const bool error_on_load_infeasible, const bool verbose) const {

        std::vector<std::pair<std::string, int>> errors;
        std::vector<std::pair<std::string, int>> warnings;

        auto inconsistent_routes = std::set<int>();

        auto customers_already_visited_in_solution = std::set<int>();

        auto total_load = 0;
        double total_cost = 0.0;

        int number_of_routes = 0;

        // Count how many times each vertex is a predecessor or successor of some other vertex.
        auto predecessor_times = std::vector<std::vector<int>>(instance.get_vertices_num());
        auto successor_times = std::vector<std::vector<int>>(instance.get_vertices_num());

        for (auto route = get_first_route(); route != Solution::dummy_route; route = get_next_route(route)) {

            number_of_routes++;

            if (is_route_empty(route)) {
                errors.emplace_back("Route " + std::to_string(route) + " is in solution but empty", __LINE__);
            }

            auto customers_already_visited_in_route = std::set<int>();

            auto initial_and_final_vertex = instance.get_depot();

            if (is_missing_depot(route)) {

                warnings.emplace_back(" Route " + std::to_string(route) +
                                          " misses the depot. It is in an inconsistent state and there is no safe way to access it "
                                          "until depot is re-inserted",
                                      __LINE__);
                inconsistent_routes.insert(route);

                // There is no simple way to access this route right now, the only way is to
                // scan vertices to find a customer belonging to this route
                for (auto c = instance.get_customers_begin(); c < instance.get_customers_end(); c++) {
                    if (customers_list[c].route_ptr == route) {
                        initial_and_final_vertex = c;
                        break;
                    }
                }
            }

            auto route_load = 0;
            double route_cost = 0.0;
            auto route_size = 0;

            auto curr = initial_and_final_vertex;
            auto next = Solution::dummy_vertex;
            do {

                // Check for double visits.
                if (customers_already_visited_in_route.count(curr)) {
                    errors.emplace_back("Vertex " + std::to_string(curr) + " in route " + std::to_string(route) +
                                            " is visited more than once within this route",
                                        __LINE__);
                    break;
                }

                // Check for double visits.
                if (curr != instance.get_depot() && customers_already_visited_in_solution.count(curr)) {
                    errors.emplace_back("Vertex " + std::to_string(curr) + " in route " + std::to_string(route) +
                                            " is visited more than once in the solution",
                                        __LINE__);
                }

                next = get_next_vertex(route, curr);
                const auto prev = get_prev_vertex(route, curr);

                // Check costs.
                if (curr == instance.get_depot()) {
                    if (std::fabs(routes_list[route].c_prev_curr - instance.get_cost(prev, curr)) > 0.01) {
                        errors.emplace_back(
                            "Vertex " + std::to_string(curr) + " in route " + std::to_string(route) + " has wrong predecessor cost",
                            __LINE__);
                    }
                } else {
                    if (std::fabs(customers_list[curr].c_prev_curr - instance.get_cost(prev, curr)) > 0.01) {
                        errors.emplace_back(
                            "Vertex " + std::to_string(curr) + " in route " + std::to_string(route) + " has wrong predecessor cost",
                            __LINE__);
                    }
                }

                // Check first customer.
                if (prev == instance.get_depot() && routes_list[route].first_customer != curr) {
                    errors.emplace_back(
                        "Vertex " + std::to_string(curr) + " in route " + std::to_string(route) +
                            " has predecessor depot but it is not the first customer of the route which is instead vertex " +
                            std::to_string(routes_list[route].first_customer),
                        __LINE__);
                }

                // Check last customer.
                if (next == instance.get_depot() && routes_list[route].last_customer != curr) {
                    errors.emplace_back("Vertex " + std::to_string(curr) + " in route " + std::to_string(route) +
                                            " has successor depot but it is not the last customer of the route which is instead vertex " +
                                            std::to_string(routes_list[route].last_customer),
                                        __LINE__);
                }

                // Check linking pointers.
                if (curr != get_prev_vertex(route, next)) {
                    errors.emplace_back("Vertex " + std::to_string(curr) + " in route " + std::to_string(route) + " has successor " +
                                            std::to_string(next) + " but the predecessor of " + std::to_string(next) +
                                            " is instead vertex " + std::to_string(get_prev_vertex(route, next)),
                                        __LINE__);
                }

                if (curr != get_next_vertex(route, prev)) {
                    errors.emplace_back("Vertex " + std::to_string(curr) + " in route " + std::to_string(route) + " has predecessor " +
                                            std::to_string(prev) + " but the successor of " + std::to_string(prev) + " is instead vertex " +
                                            std::to_string(get_next_vertex(route, prev)),
                                        __LINE__);
                }

                if (curr != instance.get_depot() && customers_list[curr].route_ptr != route) {
                    errors.emplace_back("Vertex " + std::to_string(curr) + " in route " + std::to_string(route) + " has a route pointer " +
                                            std::to_string(customers_list[curr].route_ptr),
                                        __LINE__);
                }

                predecessor_times[prev].emplace_back(curr);
                successor_times[next].emplace_back(curr);

                customers_already_visited_in_route.insert(curr);
                customers_already_visited_in_solution.insert(curr);

                route_load += instance.get_demand(curr);
                route_cost += instance.get_cost(curr, next);
                if (curr != instance.get_depot()) {
                    route_size++;
                }

                curr = next;

            } while (next != initial_and_final_vertex);

            // Return arc.
            route_cost += instance.get_cost(curr, initial_and_final_vertex);

            if (route_load != routes_list[route].load) {
                errors.emplace_back("Route " + std::to_string(route) + " has a computed load of " + std::to_string(route_load) +
                                        " but the stored one is " + std::to_string(routes_list[route].load),
                                    __LINE__);
            }

            if (route_load > instance.get_vehicle_capacity()) {
                const auto message = "Route " + std::to_string(route) + " has a load of " + std::to_string(route_load) +
                                     " but the vehicle capacity is " + std::to_string(instance.get_vehicle_capacity());
                if (error_on_load_infeasible) {
                    errors.emplace_back(message, __LINE__);
                } else {
                    warnings.emplace_back(message, __LINE__);
                }
            }

            if (route_size != routes_list[route].size) {
                errors.emplace_back("Route " + std::to_string(route) + " has a computed size of " + std::to_string(route_size) +
                                        " but the stored one is " + std::to_string(routes_list[route].size),
                                    __LINE__);
            }

            total_load += route_load;
            total_cost += route_cost;
        }

        for (auto route = get_first_route(); route != Solution::dummy_route; route = get_next_route(route)) {
            if (is_missing_depot(route)) {
                if (!routes_list[route].needs_cumulative_load_update) {
                    errors.emplace_back("Route " + std::to_string(route) + " misses the depot but has cumulative load up to date.",
                                        __LINE__);
                }
            } else {
                if (!routes_list[route].needs_cumulative_load_update) {

                    std::vector<int> load_before(routes_list[route].size, 0);
                    std::vector<int> load_after(routes_list[route].size, 0);

                    auto prev = routes_list[route].first_customer;

                    int i = 0;
                    load_before[i] = instance.get_demand(prev);
                    if (customers_list[prev].load_before != load_before[i]) {
                        errors.emplace_back(
                            "Load before mismatch at customer " + std::to_string(prev) + " of route " + std::to_string(route) + ".",
                            __LINE__);
                    }
                    load_after[i] = routes_list[route].load;
                    if (customers_list[prev].load_after != load_after[i]) {
                        errors.emplace_back(
                            "Load after mismatch at customer " + std::to_string(prev) + " of route " + std::to_string(route) + ".",
                            __LINE__);
                    }
                    ++i;

                    auto curr = customers_list[prev].next;

                    while (curr != instance.get_depot()) {
                        load_before[i] = load_before[i - 1] + instance.get_demand(curr);
                        if (customers_list[curr].load_before != load_before[i]) {
                            errors.emplace_back(
                                "Load before mismatch at customer " + std::to_string(curr) + " of route " + std::to_string(route) + ".",
                                __LINE__);
                        }
                        load_after[i] = load_after[i - 1] - instance.get_demand(prev);
                        if (customers_list[curr].load_after != load_after[i]) {
                            errors.emplace_back(
                                "Load after mismatch at customer " + std::to_string(curr) + " of route " + std::to_string(route) + ".",
                                __LINE__);
                        }
                        ++i;
                        prev = curr;
                        curr = customers_list[curr].next;
                    }
                }
            }
        }

        if (number_of_routes != depot_node.num_routes) {
            errors.emplace_back("Mismatch between stored number of routes " + std::to_string(depot_node.num_routes) +
                                    " and the actual number " + std::to_string(number_of_routes),
                                __LINE__);
        }

        if (static_cast<int>(predecessor_times[instance.get_depot()].size()) != depot_node.num_routes) {
            std::sort(predecessor_times[instance.get_depot()].begin(), predecessor_times[instance.get_depot()].end());
            std::string vertices;
            for (int vertex : predecessor_times[instance.get_depot()]) {
                vertices += std::to_string(vertex) + " ";
            }
            errors.emplace_back("Depot is predecessor of " + std::to_string(predecessor_times[instance.get_depot()].size()) +
                                    " other vertices when it should be of exactly " + std::to_string(depot_node.num_routes) +
                                    ". Depot is predecessor of " + vertices,
                                __LINE__);
        }

        if (static_cast<int>(successor_times[instance.get_depot()].size()) != depot_node.num_routes) {
            std::sort(successor_times[instance.get_depot()].begin(), successor_times[instance.get_depot()].end());
            std::string vertices;
            for (int vertex : successor_times[instance.get_depot()]) {
                vertices += std::to_string(vertex) + " ";
            }
            errors.emplace_back("Depot is successor of " + std::to_string(successor_times[instance.get_depot()].size()) +
                                    " other vertices when it should be of exactly " + std::to_string(depot_node.num_routes) +
                                    ". Depot is successor of " + vertices,
                                __LINE__);
        }

        for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
            if (predecessor_times[i].size() > 1UL) {
                std::sort(predecessor_times[i].begin(), predecessor_times[i].end());
                std::string vertices;
                for (int vertex : predecessor_times[i]) {
                    vertices += std::to_string(vertex) + " ";
                }
                errors.emplace_back("Vertex " + std::to_string(i) + " in route " + std::to_string(customers_list[i].route_ptr) +
                                        " is predecessor of " + std::to_string(predecessor_times[i].size()) +
                                        " other vertices when it should be of exactly 1. Vertex " + std::to_string(i) +
                                        " is predecessor of " + vertices,
                                    __LINE__);
            }
            if (successor_times[i].size() > 1UL) {
                std::sort(successor_times[i].begin(), successor_times[i].end());
                std::string vertices;
                for (int vertex : successor_times[i]) {
                    vertices += std::to_string(vertex) + " ";
                }
                errors.emplace_back("Vertex " + std::to_string(i) + " in route " + std::to_string(customers_list[i].route_ptr) +
                                        " is successor of " + std::to_string(successor_times[i].size()) +
                                        " other vertices  when it should be of exactly 1. Vertex " + std::to_string(i) +
                                        " is predecessor of " + vertices,
                                    __LINE__);
            }
        }

        auto customers_not_served = std::vector<int>();
        auto customers_not_served_load = 0;
        for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
            if (!is_customer_in_solution(i)) {
                customers_not_served.push_back(i);
                customers_not_served_load += instance.get_demand(i);
            } else {
                if (routes_list[customers_list[i].route_ptr].first_customer <= instance.get_depot() ||
                    routes_list[customers_list[i].route_ptr].first_customer >= instance.get_vertices_end()) {
                    errors.emplace_back("Vertex " + std::to_string(i) + " in route " + std::to_string(customers_list[i].route_ptr) +
                                            " belongs to a route for which the first customer is " +
                                            std::to_string(routes_list[customers_list[i].route_ptr].first_customer),
                                        __LINE__);
                }
                if (routes_list[customers_list[i].route_ptr].last_customer <= instance.get_depot() ||
                    routes_list[customers_list[i].route_ptr].last_customer >= instance.get_vertices_end()) {
                    errors.emplace_back("Vertex " + std::to_string(i) + " in route " + std::to_string(customers_list[i].route_ptr) +
                                            " belongs to a route for which the last customer is " +
                                            std::to_string(routes_list[customers_list[i].route_ptr].last_customer),
                                        __LINE__);
                }
                const auto route = customers_list[i].route_ptr;
                auto found = false;
                for (auto curr = get_first_customer(route); curr != instance.get_depot(); curr = get_next_vertex(curr)) {
                    if (curr == i) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    errors.emplace_back("Vertex " + std::to_string(i) + " in route " + std::to_string(customers_list[i].route_ptr) +
                                            " cannot be found by scanning the route",
                                        __LINE__);
                }
            }
        }

        if (!customers_not_served.empty()) {

            warnings.emplace_back("There are " + std::to_string(customers_not_served.size()) + " customers not served", __LINE__);
        }

        auto instance_demand_sum = 0;
        for (int i = instance.get_customers_begin(); i < instance.get_customers_end(); ++i) {
            instance_demand_sum += instance.get_demand(i);
        }
        if (total_load + customers_not_served_load != instance_demand_sum) {
            errors.emplace_back("The load of served customers is " + std::to_string(total_load) +
                                    " but the sum of the load of all customers is " + std::to_string(instance_demand_sum),
                                __LINE__);
        }

        if (std::fabs(get_cost() - total_cost) >= 0.5) {
            errors.emplace_back("The solution has a computed cost of " + std::to_string(total_cost) + " but the stored one is " +
                                    std::to_string(get_cost()),
                                __LINE__);
        }

        if (!errors.empty() || verbose) {
            std::cout << "== BEGIN OF SOLUTION FEASIBILITY CHECK REPORT ==\n";
        }

        if (!errors.empty()) {
            if (errors.size() == 1) {
                std::cout << "There is 1 error\n";
            } else {
                std::cout << "There are " << errors.size() << " errors\n";
            }
            for (const auto &entry : errors) {
                const auto message = entry.first;
                const auto line = entry.second;
                std::cout << "+ LINE " << line << " + " << message << "\n";
            }
        }

        if (!errors.empty() || verbose) {
            if (warnings.size() == 1) {
                std::cout << "There is 1 warning\n";
            } else {
                std::cout << "There are " << warnings.size() << " warnings\n";
            }
            for (const auto &entry : warnings) {
                const auto message = entry.first;
                const auto line = entry.second;
                std::cout << "+ LINE " << line << " + " << message << "\n";
            }
        }

        if (!errors.empty() || verbose) {
            std::cout << "== END OF SOLUTION FEASIBILITY CHECK REPORT ==\n";
        }

        return errors.empty();
    }


}  // namespace cobra