#include "Individual.h"

namespace genvrp {

    void Individual::evaluateCompleteCost() {
        cost = CostSol();
        for(int r = 0; r < params->data.nbVehicles; r++) {
            if(!routes[r].empty()) {
                double distance = params->data.timeCost[0][routes[r][0]];
                double load = params->data.cli[routes[r][0]].demand;
                double service = params->data.cli[routes[r][0]].serviceDuration;
                predecessors[routes[r][0]] = 0;
                for(int i = 1; i < (int)routes[r].size(); i++) {
                    distance += params->data.timeCost[routes[r][i - 1]][routes[r][i]];
                    load += params->data.cli[routes[r][i]].demand;
                    service += params->data.cli[routes[r][i]].serviceDuration;
                    predecessors[routes[r][i]] = routes[r][i - 1];
                    successors[routes[r][i - 1]] = routes[r][i];
                }
                successors[routes[r][routes[r].size() - 1]] = 0;
                distance += params->data.timeCost[routes[r][routes[r].size() - 1]][0];
                cost.distance += distance;
                cost.nbRoutes++;
                if(load > params->data.vehicleCapacity)
                    cost.capacityExcess += load - params->data.vehicleCapacity;
                if(distance + service > params->data.durationLimit)
                    cost.durationExcess += distance + service - params->data.durationLimit;
            }
        }

        cost.penalizedCost = cost.distance + cost.capacityExcess * params->ga.penaltyCapacity + cost.durationExcess * params->ga.penaltyDuration;
        isFeasible = (cost.capacityExcess < MY_EPSILON && cost.durationExcess < MY_EPSILON);
    }

    void Individual::removeProximity(Individual* indiv) {
        auto it = indivsPerProximity.begin();
        while(it->second != indiv)
            ++it;
        indivsPerProximity.erase(it);
    }

    double Individual::brokenPairsDistance(Individual* indiv2) {
        int differences = 0;
        for(int j = 1; j <= params->data.nbClients; j++) {
            if(successors[j] != indiv2->successors[j] && successors[j] != indiv2->predecessors[j])
                differences++;
            if(predecessors[j] == 0 && indiv2->predecessors[j] != 0 && indiv2->successors[j] != 0)
                differences++;
        }
        return (double)differences / (double)params->data.nbClients;
    }

    double Individual::averageBrokenPairsDistanceClosest(int nbClosest) {
        double result = 0;
        int maxSize = std::min<int>(nbClosest, indivsPerProximity.size());
        auto it = indivsPerProximity.begin();
        for(int i = 0; i < maxSize; i++) {
            result += it->first;
            ++it;
        }
        return result / (double)maxSize;
    }

    void Individual::exportCVRPLibFormat(std::string fileName) {
        std::cout << "----- WRITING SOLUTION WITH VALUE " << cost.penalizedCost << " IN : " << fileName << std::endl;
        std::ofstream myfile(fileName);
        if(myfile.is_open()) {
            for(int k = 0; k < params->data.nbVehicles; k++) {
                if(!routes[k].empty()) {
                    myfile << "Route #" << k + 1 << ":"; // Route IDs start at 1 in the file format
                    for(int i : routes[k])
                        myfile << " " << i;
                    myfile << std::endl;
                }
            }
            myfile << "Cost " << cost.penalizedCost << std::endl;
            myfile << "Time " << (double)clock() / (double)CLOCKS_PER_SEC << std::endl;
        } else
            std::cout << "----- IMPOSSIBLE TO OPEN: " << fileName << std::endl;
    }

    bool Individual::readCVRPLibFormat(std::string fileName, std::vector<std::vector<int>>& readSolution, double& readCost) {
        readSolution.clear();
        std::ifstream inputFile(fileName);
        if(inputFile.is_open()) {
            std::string inputString;
            inputFile >> inputString;
            // Loops as long as the first line keyword is "Route"
            for(int r = 0; inputString == "Route"; r++) {
                readSolution.push_back(std::vector<int>());
                inputFile >> inputString;
                getline(inputFile, inputString);
                std::stringstream ss(inputString);
                int inputCustomer;
                while(ss >> inputCustomer) // Loops as long as there is an integer to read
                    readSolution[r].push_back(inputCustomer);
                inputFile >> inputString;
            }
            if(inputString == "Cost") {
                inputFile >> readCost;
                return true;
            } else
                std::cout << "----- UNEXPECTED WORD IN SOLUTION FORMAT: " << inputString << std::endl;
        } else
            std::cout << "----- IMPOSSIBLE TO OPEN: " << fileName << std::endl;
        return false;
    }

    Individual::Individual(Params* params) : params(params) {
        successors = std::vector<int>(params->data.nbClients + 1);
        predecessors = std::vector<int>(params->data.nbClients + 1);
        routes = std::vector<std::vector<int>>(params->data.nbVehicles);
        giantTour = std::vector<int>(params->data.nbClients);
        for(int i = 0; i < params->data.nbClients; i++)
            giantTour[i] = i + 1;
        random_shuffle(giantTour.begin(), giantTour.end());
    }

    Individual::Individual() { cost.penalizedCost = 1.e30; }

} // namespace genvrp