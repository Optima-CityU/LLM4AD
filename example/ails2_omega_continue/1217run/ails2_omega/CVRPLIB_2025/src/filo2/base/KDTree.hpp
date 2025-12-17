#ifndef _FILO2_KDTREE_HPP_
#define _FILO2_KDTREE_HPP_

#include <array>
#include <queue>
#include <vector>

namespace cobra {

    // A simple implementation of a kd-tree based on https://github.com/cdalitz/kdtree-cpp.
    class KDTree {
    public:
        KDTree(const std::vector<double>& xcoords, const std::vector<double>& ycoords);
        ~KDTree();

        // Retrieves the k nearest neighbors of point (x, y).
        std::vector<int> GetNearestNeighbors(double x, double y, int k) const;

    private:
        // A 2D point representation.
        struct Point {
            Point(int index, double x, double y);
            // Point index.
            int index;
            // x and y coordinates.
            std::array<double, 2> coords;
        };

        // A KDTree node representation.
        struct Node {
            Node();
            ~Node();
            // Cut dimension: 0 for the x coordinate, 1 for the y one.
            int cutdim;
            // Child nodes.
            Node *left, *right;
            // Node bounding box.
            std::array<double, 2> lobound, hibound;
            // Index of the point in nodes vector.
            int point_index;
        };
        Node* root = nullptr;

        // Heap node representation.
        struct HeapNode {
            HeapNode(int point_index, double distance);
            // Index of the point in nodes vector.
            int point_index;
            // Distance of this neighbor from the input point.
            double distance;
        };

        // Heap node comparator.
        struct HeapNodeComparator {
            bool operator()(const HeapNode& a, const HeapNode& b) const;
        };

        // Max heap used to retrieve the set of nearest neighbors of a given point.
        using KDTreeHeap = std::priority_queue<HeapNode, std::vector<HeapNode>, HeapNodeComparator>;

        // Builds the tree using nodes indexed from begin to end excluded.
        Node* BuildTree(int depth, int begin, int end, const std::array<double, 2>& lobound, const std::array<double, 2>& hibound);

        bool SearchNeighbors(Node* tree, KDTreeHeap& heap, const std::array<double, 2>& point, int k) const;

        bool BoundsOverlapBall(const std::array<double, 2>& point, double dist, Node* node) const;
        bool BallWithinBounds(const std::array<double, 2>& point, double dist, KDTree::Node* node) const;

        std::vector<Point> nodes;
    };

}  // namespace cobra

#endif