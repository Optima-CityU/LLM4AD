#ifndef GENVRP_KMEDOIDS_H
#define GENVRP_KMEDOIDS_H

#include <algorithm>
#include <cassert>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include <map>
#include <numeric> // for std::iota (VS 2017)

#include "Clustering.h"

namespace genvrp {
    namespace kmedoids {
        using namespace clustering;

        /** Namespace containing "private" utilities. */
        namespace details {
            /** Gets the initial centres by taking k random points
             *  among all the points.
             */
            inline IndexSet getInitialCentres(std::size_t k, const DistanceMatrix& d) {
                std::random_device rd;
                std::default_random_engine rnd(rd());

                IndexSet indices(d.size());
                std::iota(indices.begin(), indices.end(), 0u);

                std::shuffle(indices.begin(), indices.end(), rnd);

                indices.erase(indices.begin() + k, indices.end());

                assert(indices.size() == k);

                return indices;
            }

            /** Gives the index of the centre nearest to a given
             *  point, together with their distance.
             */
            inline std::pair<std::size_t, double> nearestCentre(
                std::size_t pt, const IndexSet& centres, const DistanceMatrix& d)
            {
                double cost = std::numeric_limits<double>::max();
                std::size_t centre = 0u;

                for(auto i = 0u; i < centres.size(); ++i) {
                    if(d[i][pt] < cost) {
                        centre = i;
                        cost = d[i][pt];
                    }
                }

                return std::make_pair(centre, cost);
            }

            /** Given a set of centres, computes the corresponding
             *  clustering, by assigning each point to the centre
             *  to which it is closest.
             */
            inline std::pair<IndexClustering, double> getClusters(const IndexSet& centres, const DistanceMatrix& d) {
                auto clusters = IndexClustering(centres.size());
                auto total_cost = 0.0;

                for(auto pt = 0u; pt < d.size(); ++pt) {
                    const auto [c, cost] = nearestCentre(pt, centres, d);
                    clusters[c].push_back(pt);
                    total_cost += cost;
                }

                return std::make_pair(clusters, total_cost);
            }

            /** Tries to remove point c from the set of centres,
             *  and replace it with another point which is not
             *  currently a centre.
             *
             *  If any of such swaps gives an improvement in term
             *  of reducing the total cost of the clustering, it
             *  is returned (meaning: the first such swap is returned,
             *  not the best one.)
             *
             *  The cost of a clustering is the sum of all the
             *  distances between each point and its corresponding
             *  nearest centre.
             */
            inline bool trySwapCentre(
                int c, IndexSet& centres, IndexClustering& clusters, double& cost, const DistanceMatrix& d)
            {
                for(auto i = 0; i < d.size(); ++i) {
                    if(std::find(centres.begin(), centres.end(), i) != centres.end()) {
                        continue;
                    }

                    auto new_centres = centres;
                    new_centres.erase(std::remove(new_centres.begin(), new_centres.end(), c), new_centres.end());
                    new_centres.push_back(i);

                    assert(new_centres.size() == centres.size());

                    auto [new_clusters, new_cost] = getClusters(new_centres, d);

                    if(new_cost < cost) {
                        centres = new_centres;
                        clusters = new_clusters;
                        cost = new_cost;
                        return true;
                    }
                }

                return false;
            }

            /** Clusters the points into k clusters, using the
             *  greedy k-medoids algorithm.
             */
            inline IndexClustering doKmedoids(std::size_t k, const DistanceMatrix& d) {
                auto centres = getInitialCentres(k, d);
                auto [clusters, cost] = getClusters(centres, d);

                for(auto iter_num = 0u; iter_num < 100u; ++iter_num) {
                    bool improved = false;

                    for(const auto& c : centres) {
                        if(trySwapCentre(c, centres, clusters, cost, d)) {
                            improved = true;
                            break;
                        }
                    }

                    if(!improved) {
                        return clusters;
                    }
                }

                return clusters;
            }
        } // namespace details

        /** Clusters the given points into k clusters using
         *  the k-medoids algorithm.
         *
         *  Points indexed in pts_to_disregard are first removed
         *  from the set of points to cluster. After a clustering
         *  for the remaining points is found, the disregarded
         *  points are distributed equally into the given clusters.
         */
        inline IndexClustering kMedoids(std::size_t k, const DistanceMatrix& d, const IndexSet& pts_to_disregard) {
            using namespace details;

            if(k == 1u) {
                IndexCluster trivial_cluster(d.size());
                std::iota(trivial_cluster.begin(), trivial_cluster.end(), 0u);

                return {trivial_cluster};
            }

            auto all_to_clean = std::map<int, int>{};
            auto clean_to_all = std::map<int, int>{};
            int n_clean_pts = 0;

            for(auto i = 0; i < d.size(); ++i) {
                if(std::find(pts_to_disregard.begin(), pts_to_disregard.end(), i) != pts_to_disregard.end()) {
                    continue;
                } else {
                    all_to_clean[i] = n_clean_pts;
                    clean_to_all[n_clean_pts] = i;
                    ++n_clean_pts;
                }
            }

            auto cleand = DistanceMatrix(clean_to_all.size(), std::vector<double>(clean_to_all.size(), 0.0));

            for(auto i = 0u; i < clean_to_all.size(); ++i) {
                for(auto j = i + 1; j < clean_to_all.size(); ++j) {
                    cleand[i][j] = d[clean_to_all[i]][clean_to_all[j]];
                    cleand[j][i] = cleand[i][j];
                }
            }

            auto clustering = doKmedoids(k, cleand);
            auto orig_clustering = IndexClustering(k, IndexCluster{});

            assert(clustering.size() == k);

            for(auto i = 0u; i < k; ++i) {
                for(const auto& pt : clustering[i]) {
                    orig_clustering[i].push_back(clean_to_all[pt]);
                }
            }

            auto cur_id_cls = 0u;
            for(const auto& empty : pts_to_disregard) {
                orig_clustering[cur_id_cls++ % orig_clustering.size()].push_back(empty);
            }

            return orig_clustering;
        }
    } // namespace kmedoids
} // namespace genvrp


#endif //GENVRP_KMEDOIDS_H
