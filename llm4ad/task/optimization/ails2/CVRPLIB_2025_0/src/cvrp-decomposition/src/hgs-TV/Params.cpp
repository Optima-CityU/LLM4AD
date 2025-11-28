#include "Params.h"

#include <cmath>     // For std::ceil
#include <ctime>     // For std::time
#include <fstream>   // For std::ifstream, std::ofstream
#include <iostream>  // For std::cerr
#include <limits>    // For std::numeric_limits
#include <numeric>   // For std::accumulate
#include <stdexcept> // For std::invalid_argument
#include <string>
#include <unordered_set>

namespace genvrp {
    namespace {
        /** Executes a function multiple times.
         *
         *  The function can take zero params or one std::uint32_t param. In this latter case, the method will call
         *  the function, passing the iteration number as the param.
         *
         *  @tparam F           A functor type.
         *  @param how_many     How many times to repeat.
         *  @param func         The function to repeat.
         */
        template<typename F>
        inline void repeat(std::uint32_t how_many, const F& func) {
            if constexpr(std::is_invocable<F, std::uint32_t>{}) {
                for(auto i = 0u; i < how_many; ++i) {
                    func(i);
                }
            } else {
                while(how_many--) {
                    func();
                }
            }
        }

        /** Skips one or more lines when reading from stream ifs. */
        inline void skip_line(std::ifstream& ifs, std::uint32_t how_many = 1u) {
            std::string s;
            repeat(how_many, [&]() { std::getline(ifs, s); });
        }

        /** Skips reading one or more values from a stream file. */
        template<typename T>
        inline void skip_value(std::ifstream& ifs, std::uint32_t how_many = 1u) {
            T garbage;
            repeat(how_many, [&]() { ifs >> garbage; });
        }

        /** Literal operator to call repeat idiomatically:
         *    repeat(3_times, [](){ std::cout << "Hello\n"; });
         */
        std::uint32_t operator"" _times(unsigned long long x) { return x; }

        // removed unused operator<< for Params::RunStats::NewBestSource
    } // namespace

    void Params::readClients(std::ifstream& inputFile) {
        for(int i = 0; i <= data.nbClients; i++) {
            data.cli.push_back(readClient(inputFile, i));
        }
    }

    Params::Client Params::readClient(std::ifstream& inputFile, int clientNumber) {
        Client client{};

        if(type == InstanceType::Cmt) {
            inputFile >> client.custNum >> client.coordX >> client.coordY;
            inputFile >> client.serviceDuration >> client.demand;

            // Skip one integer value.
            skip_value<int>(inputFile);

            int nbPatterns;
            inputFile >> nbPatterns;

            // Skip the next nbPatterns integer values.
            skip_value<int>(inputFile, nbPatterns);
        } else if(type == InstanceType::Golden) {
            client.custNum = clientNumber;
            inputFile >> client.coordX >> client.coordY >> client.demand;
            client.serviceDuration = 0;
        } else if(type == InstanceType::Uchoa) {
            inputFile >> client.custNum >> client.coordX >> client.coordY;
            client.custNum--; // The instances count the depot as a customer.
            client.demand = 0;
            client.serviceDuration = 0;
        }

        data.maxDemand = std::max(data.maxDemand, client.demand);

        return client;
    }

    void Params::readInstance() {
        if(seed == 0) {
            std::srand(std::time(nullptr));
        } else {
            std::srand(seed);
        }

        std::ifstream inputFile{pathToInstance};

        if(inputFile.fail()) {
            throw std::invalid_argument("Invalid instance file");
        }

        data.maxDemand = 0.0;

        // Default input for CMT and Golden instances, the file type will be read from the instance
        if(type == InstanceType::Unspecified || type == InstanceType::Golden || type == InstanceType::Cmt) {
            int typeNum;
            data.isRoundingInteger = false;

            inputFile >> typeNum >> data.nbVehicles >> data.nbClients;

            skip_value<int>(inputFile);

            inputFile >> data.durationLimit >> data.vehicleCapacity;

            type = static_cast<InstanceType>(typeNum);

            if(data.durationLimit == 0) {
                data.durationLimit = std::numeric_limits<double>::max();
                data.isDurationConstraint = false;
            } else {
                data.isDurationConstraint = true;
            }
        }
        // VRP format of Uchoa et al.
        else if(type == InstanceType::Uchoa) {
            data.isRoundingInteger = true;     // The X instances of Uchoa et al. require rounding the distances
            data.isDurationConstraint = false; // There are no duration constraints for these instances
            data.durationLimit = std::numeric_limits<double>::max();

            skip_line(inputFile, 3_times);
            skip_value<std::string>(inputFile, 2_times);

            inputFile >> data.nbClients;
            --data.nbClients; // The file indicates the total number of nodes (including the depot)

            skip_line(inputFile, 2_times);
            skip_value<std::string>(inputFile, 2_times);

            inputFile >> data.vehicleCapacity;

            skip_line(inputFile, 2_times);
        }

        // Read client information
        readClients(inputFile);

        // In Uchoa's data format, the customer demands are located at the end of the file
        if(type == InstanceType::Uchoa) {
            skip_line(inputFile, 2_times);

            repeat(data.nbClients + 1, [&](auto i) {
                skip_value<int>(inputFile);
                inputFile >> data.cli[i].demand;
                data.maxDemand = std::max(data.maxDemand, data.cli[i].demand);
            });
        }

        // Calculate polar angle of each client (for the local search)
        for(int i = 0; i <= data.nbClients; i++)
            data.cli[i].polarAngle = CircleSector::positive_mod(32768. * atan2(data.cli[i].coordY - data.cli[0].coordY, data.cli[i].coordX - data.cli[0].coordX) / 3.14159265359);

        // Evaluate the total demand and minimum number of vehicles necessary for it
        data.totalDemand = std::accumulate(data.cli.begin(), data.cli.end(), 0.0, [&](double accum, const auto& cust) { return accum + cust.demand; });

        if(data.nbVehicles == -1) {
            // Default value: we give 20% more vehicles and 2 extra than the minimum given by the LB
            data.nbVehicles = std::ceil(1.2 * data.totalDemand / data.vehicleCapacity) + 2;
#ifndef BATCH_MODE
            std::cerr << "[Instance] Unknown fleet size. Using a loose bound: " << data.nbVehicles << ".\n";
#endif
        }

        data.setupDataStructures(ga.nbGranular);

        // A reasonable scale for the initial values of the penalties
        ga.penaltyDuration = 1;
        ga.penaltyCapacity = std::max(0.01, std::min(1000.0, data.maxDist / data.maxDemand));

        // If we are using a decomposition which requires the vertex frequency
        // table, initialise it.
        if(deco.requiresVfreq()) {
            vfreq = FrequencyMatrix(data.nbClients + 1, std::vector<std::size_t>(data.nbClients + 1, 0u));
        }

        // If we are using a decomposition which requires the arc frequency
        // table, initialise it.
        if(deco.requiresAfreq()) {
            afreq = FrequencyMatrix(data.nbClients + 1, std::vector<std::size_t>(data.nbClients + 1, 0u));
        }

        if(deco.requiresPfreq()) {
            pfreq = PathFrequencyMatrix();
        }
    }

    void Params::InstanceData::setupDataStructures(int nProximal) {
        // Distance matrix calculation
        maxDist = 0.0;
        timeCost = std::vector<std::vector<double>>(nbClients + 1, std::vector<double>(nbClients + 1));

        for(int i = 0; i <= nbClients; i++) {
            for(int j = 0; j <= nbClients; j++) {
                double d = std::sqrt(std::pow(cli[i].coordX - cli[j].coordX, 2) + std::pow(cli[i].coordY - cli[j].coordY, 2));

                if(isRoundingInteger) {
                    d += 0.5;
                    d = (double)(int)d; // Truncate
                }

                maxDist = std::max(maxDist, d);
                timeCost[i][j] = d;
            }
        }

        setupCorrelatedVertices(nProximal);

        // Safeguards to avoid possible numerical instability in case of instances containing
        // arbitrarily small or large numerical values
        if(maxDist < 0.1 || maxDist > 10000) {
            throw std::domain_error(
                "The distances are of very small or large scale. This could impact numerical stability. Please rescale the dataset and run again.");
        }

        if(maxDemand < 0.1 || maxDemand > 10000) {
            throw std::domain_error(
                "The load quantities are of very small or large scale. This could impact numerical stability. Please rescale the dataset and run again.");
        }
    }

    void Params::InstanceData::setupCorrelatedVertices(int nProximal) {
        // Nearest neighbors calculation
        correlatedVertices = std::vector<std::vector<int>>(nbClients + 1);
        std::vector<std::set<int>> setCorrelatedVertices(nbClients + 1);

        for(int i = 1; i <= nbClients; i++) {
            std::vector<std::pair<double, int>> proximityOrder;
            for(int j = 1; j <= nbClients; j++)
                if(i != j)
                    proximityOrder.push_back(std::pair<double, int>(timeCost[i][j], j));
            std::sort(proximityOrder.begin(), proximityOrder.end());

            for(int j = 0; j < std::min<int>(nProximal, nbClients - 1); j++) {
                // If i is correlated with j, then j should be correlated with i
                setCorrelatedVertices[i].insert(proximityOrder[j].second);
                setCorrelatedVertices[proximityOrder[j].second].insert(i);
            }
        }

        // Filling the vector of correlated vertices
        for(int i = 1; i <= nbClients; i++) {
            for(int x : setCorrelatedVertices[i]) {
                correlatedVertices[i].push_back(x);
            }
        }
    }

    std::ostream& operator<<(std::ostream& out, const Params::RunStats& stats) {
        out << stats.mpTime << "," << stats.spTime << "," << stats.nMpDecompositions << "," << stats.nMpIterations << "," << stats.nSpIterations << ","
            << stats.nRouteSpImproved << "," << stats.nRouteSpWorsened << "," << stats.nSpImproved << "," << stats.nSpWorsened;

        return out;
    }

    Params::RunStats::~RunStats() {
        if(recordBestSolutionUpdates && !newBestIndividuals.empty()) {
            std::ofstream ofs{newBestIndividualsFilename};

            if(!ofs.fail()) {
                for(const auto& bi : newBestIndividuals) {
                    ofs << bi.elapsedTime << "," << bi.cost << "\n";
                }
            } else {
                std::cerr << "Invalid output file for best individuals: " << newBestIndividualsFilename << "\n";
            }
        }
    }
} // namespace genvrp