#ifndef _FILO2_PARSER_HPP_
#define _FILO2_PARSER_HPP_

#include <optional>
#include <string>
#include <vector>

#include "../base/NonCopyable.hpp"

namespace cobra {

    // Very simple TSPLIB-like parser specialized to parse X-like instances.
    class Parser : private NonCopyable<Parser> {
    public:
        Parser(const std::string& filepath);

        // Parsed data.
        struct Data : private NonCopyable<Data> {
            // Maximum vehicle capacity.
            int vehicle_capacity;
            // Vertices x coordinates.
            std::vector<double> xcoords;
            // Vertices y coordinates.
            std::vector<double> ycoords;
            // Vertices demands.
            std::vector<int> demands;
        };

        // Parses the instance and returns the parsed data if successful, nullopt otherwise.
        std::optional<Data> Parse();

    private:
        // Instance file path.
        const std::string& filepath;
    };

}  // namespace cobra

#endif
