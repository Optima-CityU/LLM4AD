#ifndef _FILO2_SOLUTIONALGORITHMS_HPP_
#define _FILO2_SOLUTIONALGORITHMS_HPP_

#include "../base/Timer.hpp"
#include "Solution.hpp"


namespace cobra {

    // Limited savings algorithm.
    inline void clarke_and_wright(const Instance &instance, Solution &solution, const double lambda, int neighbors_num) {

        solution.reset();

        for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
            solution.build_one_customer_route</*record_acion=*/false>(i);
        }
        assert(solution.is_feasible());

        neighbors_num = std::min(instance.get_customers_num() - 1, neighbors_num);

        const auto savings_num = instance.get_customers_num() * neighbors_num;

        struct Saving {
            int i;
            int j;
            double value;
        };

        auto savings = std::vector<Saving>();
        savings.reserve(static_cast<unsigned long>(savings_num));

        for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {

            for (auto n = 1u, added = 0u; added < static_cast<unsigned int>(neighbors_num) && n < instance.get_neighbors_of(i).size();
                 n++) {

                const auto j = instance.get_neighbors_of(i)[n];

                if (i < j) {

                    const double value = +instance.get_cost(i, instance.get_depot()) + instance.get_cost(instance.get_depot(), j) -
                                         lambda * instance.get_cost(i, j);

                    savings.push_back({i, j, value});

                    added++;
                }
            }
        }


        std::sort(savings.begin(), savings.end(), [](const Saving &a, const Saving &b) { return a.value > b.value; });

#ifdef VERBOSE
        Timer timer;
#endif

        for (auto n = 0; n < static_cast<int>(savings.size()); ++n) {

            const auto &saving = savings[n];

            const auto i = saving.i;
            const auto j = saving.j;

            const auto iRoute = solution.get_route_index(i);
            const auto jRoute = solution.get_route_index(j);

            if (iRoute == jRoute) {
                continue;
            }

            if (solution.get_last_customer(iRoute) == i && solution.get_first_customer(jRoute) == j &&
                solution.get_route_load(iRoute) + solution.get_route_load(jRoute) <= instance.get_vehicle_capacity()) {

                solution.append_route(iRoute, jRoute);


            } else if (solution.get_last_customer(jRoute) == j && solution.get_first_customer(iRoute) == i &&
                       solution.get_route_load(iRoute) + solution.get_route_load(jRoute) <= instance.get_vehicle_capacity()) {

                solution.append_route(jRoute, iRoute);
            }

#ifdef VERBOSE
            if (timer.elapsed_time<std::chrono::seconds>() > 2) {
                std::cout << "Progress: " << 100.0 * (n + 1) / savings.size() << "%, Solution cost: " << solution.get_cost() << " \n";
                timer.reset();
            }
#endif
        }
        assert(solution.is_feasible());
    }

}  // namespace cobra

#endif