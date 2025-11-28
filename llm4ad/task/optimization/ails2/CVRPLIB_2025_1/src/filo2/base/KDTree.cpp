#include "KDTree.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace cobra {

    KDTree::KDTree(const std::vector<double>& xcoords, const std::vector<double>& ycoords) {

        assert(xcoords.size() == ycoords.size());

        std::array<double, 2> lobound = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
        std::array<double, 2> hibound = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};

        for (int i = 0; i < static_cast<int>(xcoords.size()); ++i) {
            lobound[0] = std::min(lobound[0], xcoords[i]);
            lobound[1] = std::min(lobound[1], ycoords[i]);
            hibound[0] = std::max(hibound[0], xcoords[i]);
            hibound[1] = std::max(hibound[1], ycoords[i]);
            nodes.emplace_back(i, xcoords[i], ycoords[i]);
        }

        root = BuildTree(0, 0, nodes.size(), lobound, hibound);
    }

    KDTree::~KDTree() {
        delete root;
    }

    KDTree::Point::Point(int index, double x, double y) : index(index), coords({x, y}) { }

    KDTree::Node::Node() = default;
    KDTree::Node::~Node() {
        delete left;
        delete right;
    }

    KDTree::HeapNode::HeapNode(int point_index, double distance) : point_index(point_index), distance(distance) { }

    bool KDTree::HeapNodeComparator::operator()(const KDTree::HeapNode& a, const KDTree::HeapNode& b) const {
        return a.distance < b.distance;
    }

    KDTree::Node* KDTree::BuildTree(int depth, int begin, int end, const std::array<double, 2>& lobound,
                                    const std::array<double, 2>& hibound) {

        const int dimension = depth % 2;

        KDTree::Node* node = new KDTree::Node();
        node->cutdim = dimension;
        node->left = nullptr;
        node->right = nullptr;
        node->lobound = lobound;
        node->hibound = hibound;

        if (end - begin <= 1) {
            node->point_index = begin;
        } else {
            int median = (begin + end) / 2;
            std::nth_element(nodes.begin() + begin, nodes.begin() + median, nodes.begin() + end,
                             [dimension](const Point& a, const Point& b) { return a.coords[dimension] < b.coords[dimension]; });
            node->point_index = median;

            const int cutval = nodes[median].coords[dimension];

            if (median - begin > 0) {
                std::array<double, 2> next_hibound = hibound;
                next_hibound[dimension] = cutval;
                node->left = BuildTree(depth + 1, begin, median, node->lobound, next_hibound);
            }

            if (end - median > 1) {
                std::array<double, 2> next_lobound = lobound;
                next_lobound[dimension] = cutval;
                node->right = BuildTree(depth + 1, median + 1, end, next_lobound, hibound);
            }
        }

        return node;
    }

    std::vector<int> KDTree::GetNearestNeighbors(double x, double y, int k) const {

        KDTreeHeap heap;

        SearchNeighbors(root, heap, {x, y}, k);

        std::vector<int> neighbors(k);

        while (!heap.empty()) {
            const HeapNode& heap_node = heap.top();
            neighbors[--k] = nodes[heap_node.point_index].index;
            heap.pop();
        }

        return neighbors;
    }

    double ComputeDistance(const std::array<double, 2>& a, const std::array<double, 2>& b) {
        return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
    }

    double ComputeCoordinateDistance(double a, double b) {
        return (a - b) * (a - b);
    }

    bool KDTree::BoundsOverlapBall(const std::array<double, 2>& point, double dist, KDTree::Node* node) const {

        double distsum = 0;

        for (int i = 0; i < static_cast<int>(point.size()); ++i) {
            if (point[i] < node->lobound[i]) {
                distsum += ComputeCoordinateDistance(point[i], node->lobound[i]);
                if (distsum > dist) return false;
            } else if (point[i] > node->hibound[i]) {
                distsum += ComputeCoordinateDistance(point[i], node->hibound[i]);
                if (distsum > dist) return false;
            }
        }

        return true;
    }

    bool KDTree::BallWithinBounds(const std::array<double, 2>& point, double dist, KDTree::Node* node) const {

        for (int i = 0; i < static_cast<int>(point.size()); ++i) {
            if (ComputeCoordinateDistance(point[i], node->lobound[i]) <= dist ||
                ComputeCoordinateDistance(point[i], node->hibound[i]) <= dist) {
                return false;
            }
        }

        return true;
    }

    bool KDTree::SearchNeighbors(KDTree::Node* node, KDTree::KDTreeHeap& heap, const std::array<double, 2>& point, int k) const {

        double currdist = ComputeDistance(point, nodes[node->point_index].coords);

        if (static_cast<int>(heap.size()) < k) {
            heap.push(HeapNode(node->point_index, currdist));
        } else if (currdist < heap.top().distance) {
            heap.pop();
            heap.push(HeapNode(node->point_index, currdist));
        }

        if (point[node->cutdim] < nodes[node->point_index].coords[node->cutdim]) {
            if (node->left) {
                if (SearchNeighbors(node->left, heap, point, k)) {
                    return true;
                }
            }
        } else {
            if (node->right) {
                if (SearchNeighbors(node->right, heap, point, k)) {
                    return true;
                }
            }
        }

        double dist = static_cast<int>(heap.size()) < k ? std::numeric_limits<double>::max() : heap.top().distance;

        if (point[node->cutdim] < nodes[node->point_index].coords[node->cutdim]) {
            if (node->right && BoundsOverlapBall(point, dist, node->right)) {
                if (SearchNeighbors(node->right, heap, point, k)) {
                    return true;
                }
            }
        } else {
            if (node->left && BoundsOverlapBall(point, dist, node->left)) {
                if (SearchNeighbors(node->left, heap, point, k)) {
                    return true;
                }
            }
        }

        if (static_cast<int>(heap.size()) == k) {
            dist = heap.top().distance;
        }

        return BallWithinBounds(point, dist, node);
    }

}  // namespace cobra