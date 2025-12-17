#ifndef GENVRP_CLUSTERING_H
#define GENVRP_CLUSTERING_H

#include <vector>
#include <cstddef>  // For std::size_t
#include <cmath>    // For std::pow

namespace genvrp {
    namespace clustering {
        /** A simple, 2-dimensional point.
         */
        struct Point2D {
            double x;
            double y;

            Point2D() = default;
            Point2D(double x, double y) : x{x}, y{y} {}
        };

        /** A structure to hold a point with an index.
         */
        struct IndexedPoint {
            Point2D pt;
            std::size_t id;

            template<typename PointT>
            IndexedPoint(PointT pt, std::size_t id) : pt{pt.x, pt.y}, id{id} {}

            explicit operator Point2D() const { return pt; }
        };

        /** A set (really, a vector) of point-like objects. */
        template<typename Point>
        using PointSet = std::vector<Point>;

        /** A cluster of points. Only semantically different from a PointSet. */
        template<typename Point>
        using Cluster = std::vector<Point>;

        /** A clustering, i.e. a set of clusters. */
        template<typename Point>
        using Clustering = std::vector<Cluster<Point>>;

        /** A set of indices of points.
         * Indices represented by ints for compatibility.
         */
        using IndexSet = std::vector<int>;

        /** A cluster expressed in terms of indices of points it contains.
         *  Indices represented by ints for compatibility.
         */
        using IndexCluster = std::vector<int>;

        /** A clustering expressed in terms of indices. */
        using IndexClustering = std::vector<IndexCluster>;

        /** A distance matrix. */
        using DistanceMatrix = std::vector<std::vector<double>>;

        /** Gets the x-coordinate of a point-like object. */
        template<typename Point>
        inline double x(const Point& pt) {
            return static_cast<Point2D>(pt).x;
        }

        /** Gets the y-coordinate of a point-like object. */
        template<typename Point>
        inline double y(const Point& pt) {
            return static_cast<Point2D>(pt).y;
        }

        /** Gets the euclidean distance squared of two point-like objects.
         *  The two objects can be of different types.
         */
        template<typename PointA, typename PointB>
        inline double distSq(const PointA& pta, const PointB& ptb) {
            return std::pow(x(pta) - x(ptb), 2) + std::pow(y(pta) - y(ptb), 2);
        }
    }
}

#endif //GENVRP_CLUSTERING_H
