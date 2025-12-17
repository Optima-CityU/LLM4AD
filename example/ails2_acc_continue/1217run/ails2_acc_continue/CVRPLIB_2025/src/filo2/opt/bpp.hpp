#ifndef _FILO2_BPP_HPP_
#define _FILO2_BPP_HPP_

#include <algorithm>
#include <vector>

#include "../instance/Instance.hpp"


namespace bpp {

    // Simple greedy solution of the bin packing problem associated with the CVRP instance.
    inline int greedy_first_fit_decreasing(const cobra::Instance& instance) {

        std::vector<int> customers(instance.get_customers_num());
        for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
            customers[i - 1] = i;
        }

        std::sort(customers.begin(), customers.end(),
                  [&instance](auto i, auto j) { return instance.get_demand(i) > instance.get_demand(j); });

        std::vector<int> bins(instance.get_customers_num(), 0);

        int used_bins = 0;
        for (auto i : customers) {
            const int demand = instance.get_demand(i);
            for (int p = 0; p < static_cast<int>(bins.size()); p++) {
                if (bins[p] + demand <= instance.get_vehicle_capacity()) {
                    bins[p] += demand;
                    if (p + 1 > used_bins) {
                        used_bins = p + 1;
                    }
                    break;
                }
            }
        }

        return used_bins;
    }

}  // namespace bpp


#endif