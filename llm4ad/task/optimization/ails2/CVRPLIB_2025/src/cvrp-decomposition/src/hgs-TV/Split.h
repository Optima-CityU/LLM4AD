#ifndef SPLIT_H
#define SPLIT_H

#include "Params.h"
#include "Individual.h"

namespace genvrp {

    struct ClientSplit {
            double demand = 0.0;
            double serviceTime = 0.0;
            double d0_x = 0.0;
            double dx_0 = 0.0;
            double dnext = 0.0;
    };

    // Simple Deque which is used for all Linear Split algorithms
    struct Trivial_Deque {
        std::vector<int> myDeque;                 // Simply a vector structure to keep the elements of the queue
        int indexFront;                           // Index of the front element
        int indexBack;                            // Index of the back element
        inline void pop_front() { indexFront++; } // Removes the front element of the queue D
        inline void pop_back() { indexBack--; }   // Removes the back element of the queue D
        inline void push_back(int i) {
            indexBack++;
            myDeque[indexBack] = i;
        } // Appends a new element to the back of the queue D
        inline int get_front() { return myDeque[indexFront]; }
        inline int get_next_front() { return myDeque[indexFront + 1]; }
        inline int get_back() { return myDeque[indexBack]; }
        void reset(int firstNode) {
            myDeque[0] = firstNode;
            indexBack = 0;
            indexFront = 0;
        }
        inline int size() { return indexBack - indexFront + 1; }

        Trivial_Deque(int nbElements, int firstNode) {
            myDeque = std::vector<int>(nbElements);
            myDeque[0] = firstNode;
            indexBack = 0;
            indexFront = 0;
        }
    };

    class Split {
    private:
        // Problem parameters
        const Params& params;
        int maxVehicles;

        /* Auxiliary data structures to run the Linear Split algorithm */
        std::vector<ClientSplit> cliSplit;
        std::vector<std::vector<double>> potential; // Potential vector
        std::vector<std::vector<int>> pred;         // Indice of the predecessor in an optimal path
        std::vector<double> sumDistance;            // sumDistance[i] for i > 1 contains the sum of distances : sum_{k=1}^{i-1} d_{k,k+1}
        std::vector<double> sumLoad;                // sumLoad[i] for i >= 1 contains the sum of loads : sum_{k=1}^{i} q_k
        std::vector<double> sumService;             // sumService[i] for i >= 1 contains the sum of service time : sum_{k=1}^{i} s_k

        // To be called with i < j only
        // Computes the cost of propagating the label i until j
        inline double propagate(int i, int j, int k) {
            return potential[k][i] + sumDistance[j] - sumDistance[i + 1] + cliSplit[i + 1].d0_x + cliSplit[j].dx_0
                   + params.ga.penaltyCapacity * std::max<double>(sumLoad[j] - sumLoad[i] - params.data.vehicleCapacity, 0.);
        }

        // Tests if i dominates j as a predecessor for all nodes x >= j+1
        // We assume that i < j
        inline bool dominates(int i, int j, int k) {
            return potential[k][j] + cliSplit[j + 1].d0_x >
                   potential[k][i] + cliSplit[i + 1].d0_x + sumDistance[j + 1] - sumDistance[i + 1] + params.ga.penaltyCapacity * (sumLoad[j] - sumLoad[i]);
        }

        // Tests if j dominates i as a predecessor for all nodes x >= j+1
        // We assume that i < j
        inline bool dominatesRight(int i, int j, int k) {
            return potential[k][j] + cliSplit[j + 1].d0_x < potential[k][i] + cliSplit[i + 1].d0_x + sumDistance[j + 1] - sumDistance[i + 1] + MY_EPSILON;
        }

        // Split for unlimited fleet
        int splitSimple(Individual* indiv);

        // Split for limited fleet
        int splitLF(Individual* indiv);

    public:
        // General Split function (tests the unlimited fleet, and only if it does not produce a feasible solution, runs the Split algorithm for limited fleet)
        void generalSplit(Individual* indiv, int nbMaxVehicles);

        // Constructor
        Split(Params& params) : params(params) {
            cliSplit = std::vector<ClientSplit>(this->params.data.nbClients + 1);
            sumDistance = std::vector<double>(this->params.data.nbClients + 1, 0.);
            sumLoad = std::vector<double>(this->params.data.nbClients + 1, 0.);
            sumService = std::vector<double>(this->params.data.nbClients + 1, 0.);
            potential = std::vector<std::vector<double>>(this->params.data.nbVehicles + 1, std::vector<double>(this->params.data.nbClients + 1, 1.e30));
            pred = std::vector<std::vector<int>>(this->params.data.nbVehicles + 1, std::vector<int>(this->params.data.nbClients + 1, 0));
        }
    };
}
#endif
