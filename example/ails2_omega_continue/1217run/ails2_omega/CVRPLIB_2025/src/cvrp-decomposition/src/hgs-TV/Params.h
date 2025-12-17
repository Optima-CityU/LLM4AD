#ifndef PARAMS_H
#define PARAMS_H

#include <vector>
#include <array>
#include <list>
#include <set>
#include <string>
#include <optional>
#include <cstddef>      // For std::size_t
#include <iostream>     // For std::cerr
#include <fstream>      // For std::ifstream
#include <sstream>
#include <algorithm>    // For std::transform
#include <unordered_set>
#include <time.h>
#include <math.h> 
#include <ext/pb_ds/assoc_container.hpp> // For __gnu_pbds
#include "CommandLineParser.h"
#include "CircleSector.h"

// Just a placeholder to have a random_shuffle in VS 2017 with the old rand() standard.
#ifdef _WIN64
template<class RandomIt>
void random_shuffle(RandomIt first, RandomIt last) {
    typename std::iterator_traits<RandomIt>::difference_type i, n;
    n = last - first;
    for(i = n - 1; i > 0; --i) {
        using std::swap;
        swap(first[i], first[std::rand() % (i + 1)]);
    }
}
#endif // _WIN64

#define MY_EPSILON 0.00001 // Precision parameter, used to avoid numerical instabilities

namespace genvrp {
    /** This class contains information which needs to be accessible from all parts of the algorithm.
     *
     *  This includes problem (instance) data and algorithm parameters.
     *  Some of this data might be changed during the execution of the algorithm, therefore a reference
     *  to the same object needs to be passed among various methods.
     */
    class Params {
    public:
        /** Instance type.
         *
         *  Unspecified => assumes that the first integer in the instance file defines the type.
         *  Cmt         => CMT instances.
         *  Golden      => Golden instances.
         *  Uchoa       => X-instances from Uchoa, or other instances with the same format.
         */
        enum InstanceType {
            Unspecified = 0,
            Cmt,
            Golden,
            Uchoa
        };

        /** Decomposition method type.
         *
         *  None        => No decomposition.
         */
        enum DecompositionType {
            None = 0,
            RandomRoute,
            BarycentreSwipe,
            BarycentreQuadrant,
            BarycentreClustering,
            RouteHistory,
            RandomArc,
            CostArc,
            ArcHistory,
            RandomPath,
            CostPath,
            PathHistory
        };

        /** Temporary structure used to represent a customer while reading the instance file. */
        struct Client {
            /** X-coordinate. */
            double coordX;
            /** Y-coordinate. */
            double coordY;
            /** Service duration (the time the vehicle needs to stop at the customer). */
            double serviceDuration;
            /** Demand (quantity to deliver at the customer). */
            double demand;
            /** Customer index, as it appears in the instance file. */
            int custNum;
            /** Polar angle of the client around the depot, measured in degrees and truncated for convenience. */
            int polarAngle;

            /** Default constructor, which zeroes all the elements. */
            Client() = default;
        };

        /** A frequency matrix is used to count how many times clients were in the same route,
         *  or how many times an arc was chosen in a good solution.
         */
        using FrequencyMatrix = std::vector<std::vector<std::size_t>>;

    private:

        /** Fixed-size path to record in the frequency matrix. */
        using RecordedPath = std::vector<int>;

        /** Access traits for the trie. */
        struct AccTraits {
            using size_type = RecordedPath::size_type;
            using key_type = RecordedPath;
            using key_const_reference = const RecordedPath&;
            using e_type = std::size_t;
            using const_iterator = RecordedPath::const_iterator;
            enum {
                max_size = 1100
            };

            inline static const_iterator begin(key_const_reference k) { return k.begin(); }
            inline static const_iterator end(key_const_reference k) { return k.end(); }
            inline static std::size_t e_pos(e_type e) { return e; }
        };

    public:

        /** Structure mapping paths to frequencies. */
        using PathFrequencyMatrix = __gnu_pbds::trie<RecordedPath, std::uint32_t, AccTraits>;

    private:

        /** Algorithm parameters. */
        struct GeneticParameters {
            /** Penalty coefficient for one unit of capacity violation. */
            double penaltyCapacity = 0.01;
            /** Penalty coefficient for one unit of distance violation. */
            double penaltyDuration = 1.0;
            /** Target number of feasible individuals. */
            double targetFeasible = 0.3;
            /** Programme timeout, in seconds. */
            double timeoutSec = 3600.0;
            /** Granular search parameter. */
            int nbGranular = 20;
            /** Number of closest individuals considered in the diversity contribution metric. */
            int nbCountDistMeasure = 5;
            /** Number of elite individuals in the population. */
            int nbElite = 4;
            /** Minimum population size. */
            int mu = 25;
            /** Number of solutions created before reaching the maximum population size. */
            int lambda = 40;
            /** Number of iterations between the penalty multiplier adjustments. */
            int adjMultiplierIters = 100;
            /** Number of iterations between status reports to stdout. */
            int outputIters = 500;
            /** Maximum number of iterations of the algorithm. */
            int maxIter = 100'000'000;
            /** Maximum number of iterations without improvement. */
            int maxIterNonProd = 20'000; // TV -- I set it to allow restarts of the GA in the level-0 decomposition, instead of looping forever and wasting the alloted time

            /** Default constructor, which initialises all elements with their default values. */
            GeneticParameters() = default;

            /** Copy constructor, performing a member-wise copy. */
            GeneticParameters(const GeneticParameters&) = default;
        };

        /** Parameters related to decomposition. */
        struct DecompositionParameters {
            /** Fraction of individuals to consider "elite" for decomposition. */
            double eliteFraction = 0.25;

            /** Decomposition type. */
            DecompositionType type = DecompositionType::None;

            /** Number of iterations between two decompositions. */
            int decompositionIters = 500;

            /** Maximum number of customers in decomposed problems, when it can be limited. */
            int targetMaxSpCustomers = 200;

            /** Number of iterations the subproblem GA runs. */
            int spMaxIter = 10000;

            /** Number of iterations without improvement to stop the subproblem GA. */
            int spMaxIterNonProd = 1000;

            /** Number of elite individuals in the subproblem GA's population. */
            int spNbElite = 2;

            /** Parameter lambda to use in the subproblem. */
            int spMu = 12;

            /** Parameter mu to use in the subproblem. */
            int spLambda = 20;

            /** Maximum decomposition recursion level. 0 is the MP only, from 1 are the SPs. */
            int maxDecoLevel = 3;

            /** If true, the subproblem is warm-started with the master problem elite solution. */
            bool warmStart= true;

            /** Default constructor, which initialises all elements with their default values. */
            DecompositionParameters() = default;

            /** Copy constructor, performing a member-wise copy. */
            DecompositionParameters(const DecompositionParameters&) = default;

            /** Tells if the decomposition method requires a clients frequency table. */
            bool requiresVfreq() const { return type == DecompositionType::RouteHistory; }

            /** Tells if the decomposition method requires an arc frequency table. */
            bool requiresAfreq() const { return type == DecompositionType::ArcHistory; }

            /** Tells if the decomposition method requires a path frequency table. */
            bool requiresPfreq() const { return type == DecompositionType::PathHistory; }
        };

        /** Problem data for the instance being solved. */
        struct InstanceData {
            /** Vector containing information on each customer.
             *
             *  It contains nbClients + 1 elements, where 0 is actually the depot.
             *  Real customers start from index 1.
             */
            std::vector<Client> cli;

            /** Distance matrix.
             *
             *  It has size (nbCustomers + 1) x (nbCustomers + 1).
             */
            std::vector<std::vector<double>> timeCost;

            /** For each vertex, list of nearby vertices.
             *
             *  It is a vector of nbClients + 1 elements.
             *  Each vector contains at most GeneticParameters::nbGranular customers, in increasing order of distance.
             */
            std::vector<std::vector<int>> correlatedVertices;

            /** Sum of the demands of all clients. */
            double totalDemand;

            /** Route duration limit. */
            double durationLimit;

            /** Vehicle capacity limit. */
            double vehicleCapacity;

            /** Highest demand, among all customers. */
            double maxDemand;

            /** Largest distance between two vertices. */
            double maxDist;

            /** Number of customers in the instance (the depot is not counted). */
            int nbClients;

            /** Number of available vehicles. */
            int nbVehicles;

            /** Knapsack-style lower bound on the number of vehicles needed to satisfy the demand. */
            int lbVehicles;

            /** True if the problem type requires distances to be rounded to the nearest integer. */
            bool isRoundingInteger;

            /** True if the problem type has maximum route duration constraints. */
            bool isDurationConstraint;

            /** Default constructor, which zeroes or otherwise default-constructs all data members. */
            InstanceData() = default;

            /** Default copy constructor, performing member-wise copies. */
            InstanceData(const InstanceData&) = default;

            /** Calculates #maxDist, #timeCost, and #correlatedVertices, starting from the other data. */
            void setupDataStructures(int nProximal);

            /** Recomputes the correlated vertices structures. */
            void setupCorrelatedVertices(int nProximal);
        };

    public:

        /** Satistics collected during the run. */
        struct RunStats {
            /** Tells which part of the algorithm produced a new best individual. */
            enum NewBestSource {
                MainGenetic = 0, // Main genetic components (xo, ls, ...)
                Decomposition    // Subproblem from decomposition
            };

            /** Contains info on updates to the best individual. */
            struct NewBest {
                NewBestSource source;
                double cost;
                double elapsedTime;
            };

            /** Time spent solving the master problem. */
            double mpTime = 0.0;

            /** Time spent solving the subproblem. */
            double spTime = 0.0;

            /** Number of times the master problem was decomposed. */
            std::size_t nMpDecompositions = 0;

            /** Number of MP iterations. */
            std::size_t nMpIterations = 0;

            /** Overall number of SP iterations. */
            std::size_t nSpIterations = 0;

            /** Number of times a single subproblem gave a better solution than its piece of starting mp solution.
             *  Only makes sense for route-based decomposition.
             */
            std::size_t nRouteSpImproved = 0;

            /** Number of times a single subproblem gave a worse solution than its piece of starting mp solution.
             *  Only makes sense for route-based decomposition.
             */
            std::size_t nRouteSpWorsened = 0;

            /** Number of times the subproblems gave a better solution than the elite starting one. */
            std::size_t nSpImproved = 0;

            /** Number of times the subproblems gave a worse solution than the elite starting one. */
            std::size_t nSpWorsened = 0;

            /** Whether to record data on updates to the best feasible individual. */
            bool recordBestSolutionUpdates = true;

            /** List of updates to the best feasible individual. */
            std::vector<NewBest> newBestIndividuals = {};

            /** Where to save data about updates to the best feasible individual. */
            std::string newBestIndividualsFilename = "nest-individuals.txt";

            /** Writes best individuals data to file, if enabled. */
            ~RunStats();
        };

    private:

        /** Reads the list of clients from contiguous rows in the input file and fills #cli.
         *
         *  Requires the input file stream's pointer to be just before the customers' section starts.
         *
         *  @param inputFile    An open input file stream containing instance data.
         */
        void readClients(std::ifstream& inputFile);

        /** Reads data for one client from the input file.
         *
         *  Requires the input file stream's pointer to be just before the #clientNumber -th client data.
         *
         *  @param inputFile        An open input file stream containing instance data.
         *  @param clientNumber     The client number, in the order given in the instance file.
         *  @return                 An object representing the customer read from file.
         */
        Client readClient(std::ifstream& inputFile, int clientNumber);

    public:

        /** Length (in term of nodes) of path to record in the path frequency matrix.
         *
         *  Note: record_path_size = 1 does not make sense,
         *        record_path_size = 2 is just arcs.
         */
        static constexpr std::size_t record_path_size = 3u;

        /** Genetic algorithm parameters. */
        GeneticParameters ga;

        /** Decomposition method parameters. */
        DecompositionParameters deco;

        /** Problem instance data. */
        InstanceData data;

        /** Run statistics. */
        RunStats stats;

        /** Path to the instance file. */
        std::string pathToInstance;

        /** Path to the file that will contain the solution. */
        std::string pathToSolution;

        /** Vertex frequency matrix (how many times two vertices were in the same route?)
         *  This is only used if some decomposition method requiring it is active.
         */
        std::optional<FrequencyMatrix> vfreq = std::nullopt;

        /** Arc frequency matrix (how many times an arc was chosen in a good solution?)
         *  This is only used if some decomposition method requiring it is active.
         */
        std::optional<FrequencyMatrix> afreq = std::nullopt;

        /** Path frequency matrix (how many times a path of a fixed size was contained
         *  in a good solution?). This is only needed if some decomposition method
         *  requiring it is active.
         */
        std::optional<PathFrequencyMatrix> pfreq = std::nullopt;

        /** Problem type. */
        InstanceType type = InstanceType::Unspecified;

        /** Random seed. */
        unsigned int seed = 0u;

        /** Move constructor, to build an object from a temporary r-value reference. */
        Params(Params&&) = default;

        /** Default copy constructor. */
        Params(const Params&) = default;

        /** Setups the correlated vertices data. */
        void setupDataCorrelatedVertices() { data.setupCorrelatedVertices(ga.nbGranular); }

        /** Prints the running stats. */
        friend std::ostream& operator<<(std::ostream& out, const Params::RunStats& stats);

    private:

        /** Default constructor. Private to be only used by the factory class. */
        Params() = default;

        /** Reads data from the instance file. */
        void readInstance();

        /** Factory class to be used to create Params objects. */
        friend class ParamsFactory;
    };

    /** Factory class to be used to create Params objects. */
    class ParamsFactory {

        /** The Params object we will build and finally return to the user. */
        Params p;

    public:

        /** Start by default-initialising the parameters. */
        ParamsFactory() : p{} {}

        /** Fill in part of the params object with data from the command line. */
        ParamsFactory& withCommandLine(const CommandLineParser& c) {
            withInstanceFile(c.getInstancePath());
            withSolutionFile(c.getSolutionPath());
            withTimeoutSec(c.getTimeoutSec());
            withInstanceTypeStr(c.getType());
            withNbVehicles(c.getNbVehicles());
            withRandomSeed(c.getRandomSeed());
            withDecompositionTypeStr(c.getDecoType());
            withDecoIters(c.getDecoIterations());
            withSpSize(c.getSubproblemSz());
            withWarmStart(c.getWarmStart());
            withBestIndividualsPath(c.getBssPath());

            return *this;
        }

        /** Sets the instance file path in the params. */
        ParamsFactory& withInstanceFile(std::string iFile) {
            p.pathToInstance = std::move(iFile);
            return *this;
        }

        /** Sets the solution file path in the params. */
        ParamsFactory& withSolutionFile(std::string sFile) {
            p.pathToSolution = std::move(sFile);
            return *this;
        }

        /** Sets the timeout in the params. */
        ParamsFactory& withTimeoutSec(double t) {
            p.ga.timeoutSec = t;
            return *this;
        }

        /** Sets the random seed in the params. */
        ParamsFactory& withRandomSeed(int seed) {
            p.seed = seed;
            return *this;
        }

        /** Sets the instance type in the params. It accepts both long-form and numeric-form types. */
        ParamsFactory& withInstanceTypeStr(std::string type) {
            std::transform(type.begin(), type.end(), type.begin(), ::tolower);

            if(type == "cmt" || type == "1") {
                p.type = Params::InstanceType::Cmt;
            } else if(type == "golden" || type == "2") {
                p.type = Params::InstanceType::Golden;
            } else if(type == "uchoa" || type == "3") {
                p.type = Params::InstanceType::Uchoa;
            } else {
                std::cerr << "Warning: type not recognised: " << type << "\n";
                p.type = Params::InstanceType::Unspecified;
            }

            return *this;
        }

        /** Sets the decomposition type in the params. */
        ParamsFactory& withDecompositionTypeStr(std::string type) {
            std::transform(type.begin(), type.end(), type.begin(), ::tolower);

            if(type == "randomroute") {
                p.deco.type = Params::DecompositionType::RandomRoute;
            } else if(type == "barycentreswipe" || type == "barycenterswipe") {
                p.deco.type = Params::DecompositionType::BarycentreSwipe;
            } else if(type == "barycentrequadrant" || type == "barycenterquadrant") {
                p.deco.type = Params::DecompositionType::BarycentreQuadrant;
            } else if(type == "barycentreclustering" || type == "barycenterclustering") {
                p.deco.type = Params::DecompositionType::BarycentreClustering;
            } else if(type == "routehistory") {
                p.deco.type = Params::DecompositionType::RouteHistory;
            } else if(type == "randomarc") {
                p.deco.type = Params::DecompositionType::RandomArc;
            } else if(type == "costarc") {
                p.deco.type = Params::DecompositionType::CostArc;
            } else if(type == "archistory") {
                p.deco.type = Params::DecompositionType::ArcHistory;
            } else if(type == "randompath") {
                p.deco.type = Params::DecompositionType::RandomPath;
            } else if(type == "costpath") {
                p.deco.type = Params::DecompositionType::CostPath;
            } else if(type == "pathhistory") {
                p.deco.type = Params::DecompositionType::PathHistory;
            } else {
                #ifndef BATCH_MODE
                    std::cerr << "[Algorithm] Decomposition type not recognised: " << type << "; disabling decomposition.\n";
                #endif
                p.deco.type = Params::DecompositionType::None;
            }

            return *this;
        }

        /** Sets the number of vehicles in the params. */
        ParamsFactory& withNbVehicles(int n) {
            p.data.nbVehicles = n;
            return *this;
        }

        /** Sets the target subproblem size. */
        ParamsFactory& withSpSize(int n) {
            p.deco.targetMaxSpCustomers = n;
            return *this;
        }

        /** Sets the number of iterations between decompositions. */
        ParamsFactory& withDecoIters(int n) {
            p.deco.decompositionIters = n;
            return *this;
        }

        /** Activates warm-starting the subproblems. */
        ParamsFactory& withWarmStart(bool w) {
            p.deco.warmStart = w;
            return *this;
        }

        /** Activate recording data for best individuals. */
        ParamsFactory& withBestIndividualsPath(std::string path) {
            if(!path.empty()) {
                p.stats.recordBestSolutionUpdates = true;
                p.stats.newBestIndividualsFilename = path;
            }

            return *this;
        }

        /** Returns the params object held. This puts #p in a moved-from state. */
        Params&& get() {
            p.readInstance();
            return std::move(p);
        }
    };
}
#endif