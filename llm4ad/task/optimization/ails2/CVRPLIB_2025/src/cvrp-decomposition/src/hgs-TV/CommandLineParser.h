#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H

#include <iostream>
#include <string>
#include <cassert>
#include <cstdlib>      // For atoi

namespace genvrp {
    /** Utility class to parse command-line parameters. */
    class CommandLineParser {
        /** Instance type. For example "Golden", or "Uchoa". */
        std::string type;

        /** Decomposition type. For example "None" of "BarycentreRandom". */
        std::string decoType;

        /** Path to the instance file. */
        std::string instancePath;

        /** Path to the file where the solution should be saved. */
        std::string outputPath;

        /** Path to the file where info on best solutions should be saved. */
        std::string bssPath;

        /** Algorithm timeout in second. */
        int timeoutSec;

        /** Random number generator seed. */
        int randomSeed;

        /** User-provided number of vehicles to use. */
        int nbVehicles;

        /** Target subproblem size. */
        int subproblemSz;

        /** Number of iterations between two decomposition phases. */
        int decoIterations;

        /** True if subproblems should be warmstarted. */
        bool warmStart;

        /** True iff the command-line parameters provided are valid. */
        bool commandOk;

        /** Sets the default output file name.
         *
         *  The output file is set to be in the same directory as the instance file.
         *  Furthermore, the file name is given by concatenating string "sol-" and the filename of the instance file.
         */
        void setDefaultOutputName();

        /** Prints a help message. */
        void displayHelp() const;

        /** Parses command line arguments and sets the #commandOk flag. */
        void parseCommandLine(int argc, char* argv[]);

    public:

        /** Parses the command line arguments and checks their validity. */
        CommandLineParser(int argc, char *argv[]) {
            parseCommandLine(argc, argv);
        }

        std::string getInstancePath() const { assert(commandOk); return instancePath; }
        std::string getSolutionPath() const { assert(commandOk); return outputPath; }
        std::string getDecoType() const { assert(commandOk); return decoType; }
        std::string getType() const { assert(commandOk); return type; }
        std::string getBssPath() const { assert(commandOk); return bssPath; }
        int getTimeoutSec() const { assert(commandOk); return timeoutSec; }
        int getNbVehicles() const { assert(commandOk); return nbVehicles; }
        int getRandomSeed() const { assert(commandOk); return randomSeed; }
        int getSubproblemSz() const { assert(commandOk); return subproblemSz; }
        int getDecoIterations() const { assert(commandOk); return decoIterations; }
        bool getWarmStart() const { assert(commandOk); return warmStart; }
        bool isValid() const { return commandOk; }
    };
}
#endif
