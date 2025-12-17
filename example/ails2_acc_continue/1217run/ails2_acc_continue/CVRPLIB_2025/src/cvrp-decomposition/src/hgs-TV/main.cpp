#include "CommandLineParser.h"
#include "Genetic.h"
#include "Params.h"

#include <iostream> // For std::cout

int main(int argc, char* argv[]) {
    using namespace genvrp;

    CommandLineParser c{argc, argv};

    if(c.isValid()) {
        Params params = ParamsFactory{}.withCommandLine(c).get();

#ifndef BATCH_MODE
        std::cout << "[Algorithm] Loaded instance with " << params.data.nbClients << " customers and " << params.data.nbVehicles << " vehicles.\n";
        std::cout << "[Algorithm] Building the initial population.\n";
#endif

        Genetic solver{params};

        solver.run();

#ifndef BATCH_MODE
        solver.exportBestSolution();
        std::cout << "[Algorithm] Algorithm terminated.\n";
#else
        solver.printBestSolutionValue();
#endif
    }

    return 0;
}
