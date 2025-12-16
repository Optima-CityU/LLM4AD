#include "CommandLineParser.h"

namespace genvrp {
    namespace {
        // To use when std::filesystem::path::replace_filename is not available.
        // Of course it is not portable, etc.
        // Cave canem!
        std::string add_prefix_to_filename(const std::string& path, const std::string& prefix) {
            const auto last_slash = path.find_last_of('/');
            const auto dir = path.substr(0, last_slash + 1);
            const auto file = path.substr(last_slash + 1);
            return dir + prefix + file;
        }
    } // namespace

    void CommandLineParser::setDefaultOutputName() { outputPath = add_prefix_to_filename(instancePath, "sol-"); }

    void CommandLineParser::displayHelp() const {
        std::cout << "genvrp: Solves (Capacitated) Vehicle Routing Problems via a hybrid genetic algorithm.\n\n"
                  << "Usage: genvrp [-? | -help]\n"
                  << "Usage: genvrp instancePath [OPTIONS]\n\n";
        std::cout << "OPTIONS:\n\n";
        std::cout << "\t-sol solPath\n"
                  << " \t\tSets the output solution path.\n"
                  << "\t\tDefault is the instance file name prepended with sol-, in the same directory\n"
                  << "\t\twhere the instance file is.\n\n";
        std::cout << "\t-t timeout\n"
                  << "\t\tSets the timeout in seconds.\n"
                  << "\t\tDefault is 360 seconds (6 minutes).\n\n";
        std::cout << "\t-seed rndSeed\n"
                  << "\t\tSets the random seed, for reproducible runs.\n"
                  << "\t\tIf the seed is 0, the RNG is seeded with the current time.\n"
                  << "\t\tDefault is 0.\n\n";
        std::cout << "\t-veh nbVehicles\n"
                  << "\t\tSets the fleet size.\n"
                  << "\t\tIt is not necessary for instances of type CMT and Golden, because this information\n"
                  << "\t\tis already given in the instance file. It is also not necessary for instances of\n"
                  << "\t\ttype Uchoa in the original set, as we have a list of well-known fleet sizes. If\n"
                  << "\t\tthe user uses an instance of type Uchoa which is not in the original set, then this\n"
                  << "\t\tparameter has to be specified. Otherwise, the algorithm will use a lax upper bound\n"
                  << "\t\ton the number of vehicles needed, computed from demand and vehicle capacity.\n\n";
        std::cout << "\t-type instanceType\n"
                  << "\t\tSets the instance type.\n"
                  << "\t\tAllowed types are 'CMT', 'Golden', and 'Uchoa'. If no type is given, the programme\n"
                  << "\t\ttries to read it from the instance file.\n\n";
        std::cout << "\t-deco decoType\n"
                  << "\t\tSets the type of decomposition.\n"
                  << "\t\tAllowed types are 'None', 'RandomRoute', 'BarycentreClustering', 'BarycentreQuadrant',\n"
                  << "\t\t'BarycentreSwipe', 'RouteHistory', 'RandomArc', 'CostArc', 'ArcHistory', 'RandomPath',\n"
                  << "\t\t'CostPath', and 'PathHistory'.\n"
                  << "\t\tDefault is 'None'.\n\n";
        std::cout << "\t-sz subproblemSz\n"
                  << "\t\tSets the target subproblem size when decomposing.\n"
                  << "\t\tDefault is 200.\n\n";
        std::cout << "\t-di decoIters\n"
                  << "\t\tSets the number of iterations between decomposition phases.\n"
                  << "\t\tDefault is 10000.\n\n";
        std::cout << "\t-ws warmStart\n"
                  << "\t\tPass 1 to warm-start the decomposed subproblems with the master problem solution.\n"
                  << "\t\tDefault is 1.\n\n";
        std::cout << "\t-bss fileName\n"
                  << "\t\tIf passed, it contains the filename where to store data about the best solutions found.\n\n";
    }

    void CommandLineParser::parseCommandLine(int argc, char* argv[]) {
        commandOk = true;

        if(argc % 2 != 0 || argc < 2) {
            std::cout << "[CommandLine] Wrong number of parameters: " << argc << "\n";
            commandOk = false;
        } else if(std::string(argv[1]) == "-?" || std::string(argv[1]) == "-help") {
            commandOk = false;
        } else {
            instancePath = std::string(argv[1]);
            setDefaultOutputName();

            timeoutSec = 360;
            randomSeed = 0;
            type = "";
            decoType = "";
            bssPath = "";
            nbVehicles = -1;
            warmStart = true;
            subproblemSz = 200;
            decoIterations = 10000;

            for(int i = 2; i < argc; i += 2) {
                if(std::string(argv[i]) == "-t") {
                    timeoutSec = std::atoi(argv[i + 1]);
                } else if(std::string(argv[i]) == "-sol") {
                    outputPath = std::string(argv[i + 1]);
                } else if(std::string(argv[i]) == "-seed") {
                    randomSeed = std::atoi(argv[i + 1]);
                } else if(std::string(argv[i]) == "-type") {
                    type = argv[i + 1];
                } else if(std::string(argv[i]) == "-veh") {
                    nbVehicles = std::atoi(argv[i + 1]);
                } else if(std::string(argv[i]) == "-deco") {
                    decoType = argv[i + 1];
                } else if(std::string(argv[i]) == "-ws") {
                    warmStart = (std::string(argv[i + 1]) == "1");
                } else if(std::string(argv[i]) == "-sz") {
                    subproblemSz = std::atoi(argv[i + 1]);
                } else if(std::string(argv[i]) == "-di") {
                    decoIterations = std::atoi(argv[i + 1]);
                } else if(std::string(argv[i]) == "-bss") {
                    bssPath = argv[i + 1];
                } else {
                    std::cout << "[CommandLine] Unrecognised parameter: " << argv[i] << "\n";
                    commandOk = false;
                }
            }
        }

        if(!commandOk) {
            displayHelp();
        }
    }
} // namespace genvrp