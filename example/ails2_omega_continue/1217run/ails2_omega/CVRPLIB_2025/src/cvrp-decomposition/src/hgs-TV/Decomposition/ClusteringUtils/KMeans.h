#ifndef GENVRP_KMEANS_H
#define GENVRP_KMEANS_H

#include "Clustering.h"

#include <vector>
#include <cassert>
#include <algorithm> // For std::transform
#include <random>    // For std::rand, RAND_MAX
#include <cstddef>   // For std::size_t
#include <numeric>   // For std::accumulate (VS 2017)

namespace genvrp {
    namespace kmeans {
        using namespace clustering;

        /** Namespace containing "private" utilities. */
        namespace detail {
            /** Return the index of an element in a vector of doubles, chosen
             *  with probability proportional with the value held.
             */
            inline std::size_t rouletteWheelSelection(const std::vector<double>& weights) {
                // Calculate the sum of all weights in the vector
                const auto sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);

                // Pivot value used to choose the index (between 0 and sum_weights).
                const auto pivot = ((double)std::rand() / RAND_MAX) * sum_weights;
                auto cur_sum = 0.0;

                // Accumulate values until you reach the pivot.
                for(auto i = 0u; i < weights.size() - 1u; ++i) {
                    cur_sum += weights[i];

                    if(cur_sum >= pivot) {
                        return i;
                    }
                }

                // If the pivot was not reached, then the chosen index must be the largest one.
                return weights.size() - 1u;
            }

            /** Gives the minimum distance between a point-like object and a set
             *  of other point-like objects.
             */
            template<typename PointA, typename PointB>
            inline double minDistsq(const PointA& p, const PointSet<PointB>& centres) {
                auto m = std::numeric_limits<double>::max();

                for(const auto& c : centres) {
                    const auto d = distSq(p, c);
                    if(d < m) {
                        m = d;
                    }
                }

                assert(m < std::numeric_limits<double>::max());

                return m;
            }

            /** Uses the k-means++ algorithm for initialising the cluster centres
             *  at the beginning of the k-means algorithm.
             *
             *  Reference: http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
             */
            template<typename Point>
            inline PointSet<Point2D> getInitialCentres(std::size_t k, PointSet<Point> pts) {
                PointSet<Point2D> centres;
                centres.reserve(k);

                // The first centre is simply taking at random among the points.
                const auto init_pt = std::rand() % pts.size();
                centres.push_back(static_cast<Point2D>(pts[init_pt]));
                pts.erase(pts.begin() + init_pt);

                // Repeat until we have k centres.
                while(centres.size() < k) {
                    // For each point, compute the distance between the point and the
                    // nearest centre which has been chosen until now.
                    std::vector<double> distances;
                    std::transform(pts.begin(), pts.end(), std::back_inserter(distances),
                                   [&centres](const auto& p) -> double { return minDistsq(p, centres); });

                    // Choose the new centre with probability proportional to
                    // the distances computed.
                    const auto new_pt = rouletteWheelSelection(distances);
                    centres.push_back(static_cast<Point2D>(pts[new_pt]));
                    pts.erase(pts.begin() + new_pt);
                }

                assert(centres.size() == k);
                return centres;
            }

            /** Tells whether two set of point-like objects differ significantly component-wise,
             *  i.e. the coordinates of the respective points are different of more than a
             *  threshold. (In our case, 1e-2).
             */
            template<typename PointA, typename PointB>
            inline bool different(const PointSet<PointA>& v1, const PointSet<PointB>& v2) {
                assert(v1.size() == v2.size());

                for(auto i = 0u; i < v1.size(); ++i) {
                    if(std::abs(x(v1[i]) - x(v2[i])) > 1e-2 && std::abs(y(v1[i]) - y(v2[i])) > 1e-2) {
                        return true;
                    }
                }

                return false;
            }

            /** Calculate the centres of each cluster of a clustering, as the
             *  the centre of mass of the clusters.
             */
            template<typename Point>
            inline PointSet<Point2D> getCentres(const Clustering<Point>& clusters) {
                PointSet<Point2D> centres;
                centres.reserve(clusters.size());

                for(const auto& cl : clusters) {
                    double avg_x = 0.0, avg_y = 0.0;

                    for(const auto& pt : cl) {
                        avg_x += x(pt);
                        avg_y += y(pt);
                    }

                    centres.emplace_back(avg_x / cl.size(), avg_y / cl.size());
                }

                return centres;
            }

            /** Gives the index of the centre (out of those contained in centres)
             *  which is closest to the given point pt.
             */
            template<typename Point>
            inline std::size_t nearestCentre(const Point& pt, const PointSet<Point2D>& centres) {
                std::size_t n = centres.size() - 1u;
                double dsq = distSq(pt, centres.back());

                for(auto i = 0u; i < centres.size() - 1u; ++i) {
                    if(distSq(pt, centres[i]) < dsq) {
                        n = i;
                    }
                }

                return n;
            }

            /** Returns a clustering by assigning each point to the cluster corresponding
             *  to the centre which is nearest to the given point.
             */
            template<typename Point>
            inline Clustering<Point> kmeansClusteringAssignment(const PointSet<Point>& pts,
                                                                const PointSet<Point2D>& centres)
            {
                Clustering<Point> clusters(centres.size());

                for(const auto& pt : pts) {
                    clusters[nearestCentre(pt, centres)].push_back(pt);
                }

                return clusters;
            }
        } // namespace detail

        /** Gives a clustering of pts made up of k clusters, using the k-means algorithm.
         *
         *  Parameter pts_to_disregard contains indices of points to remove from pts before
         *  the clustering algorithm starts. These points will be reinserted at the end
         *  distributing them equally (in a balanced way) among the resulting clusters.
         *  In our application these points correspond to empty routes (see comments inside
         *  the function body).
         */
        template<typename PointT>
        inline IndexClustering kMeans(std::size_t k, const PointSet<PointT> pts, const IndexSet& pts_to_disregard) {
            using namespace detail;

            // For k = 1, just give back a cluster with all nodes.
            if(k == 1u) {
                IndexCluster trivial_cluster(pts.size());
                std::iota(trivial_cluster.begin(), trivial_cluster.end(), 0);

                return {trivial_cluster};
            }

            // Create a set of points from pts, removing those to disregard.
            auto ipts = PointSet<IndexedPoint>{};
            for(auto i = 0u; i < pts.size(); ++i) {
                if(std::find(pts_to_disregard.begin(), pts_to_disregard.end(), i) == pts_to_disregard.end()) {
                    ipts.emplace_back(pts[i], i);
                }
            }

            // Initialise k-means by choosing the initial centres (with k-means++).
            auto centres = getInitialCentres(k, ipts);
            // Get the initial clustering by nearest distance.
            auto clusters = kmeansClusteringAssignment(ipts, centres);

            // To avoid numerical stability problems, we limit the number of
            // iterations to 100. Therefore, the algorithm will stop when either
            // the centres don't move anymore, or 100 iterations passed.
            for(auto iter_num = 0u; iter_num < 100u; ++iter_num) {
                // Compute the new centres as the centres of mass
                // of the current clusters.
                const auto new_centres = getCentres(clusters);

                // If the centres moved...
                if(different(new_centres, centres)) {
                    // Update the centres and recompute the clusters
                    centres = new_centres;
                    clusters = kmeansClusteringAssignment(ipts, centres);
                } else {
                    // Otherwise, exit.
                    break;
                }
            }

            // Build an index clustering from the k-means clustering.
            auto idcls = IndexClustering{};
            for(const auto& cl : clusters) {
                idcls.emplace_back();

                for(const auto& ipt : cl) {
                    idcls.back().push_back(ipt.id);
                }
            }

            // We now have a clustering, but what to do with the
            // points to disregard? Because in our application they
            // correspond to empty routes (therefore, "free" vehicles)
            // and we don't know where to place them, a priori, we
            // simply try to distribute them uniformly amonng the k
            // clusters.
            auto cur_id_cls = 0u;
            for(const auto& empty : pts_to_disregard) {
                idcls[cur_id_cls++ % idcls.size()].push_back(empty);
            }

            return idcls;
        }
    } // namespace kmeans
}

#endif //GENVRP_KMEANS_H
