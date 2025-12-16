#include "Parser.hpp"

#include <cstring>

namespace cobra {

    Parser::Parser(const std::string& filepath) : filepath(filepath) { }

    std::optional<Parser::Data> Parser::Parse() {

        FILE* file = fopen(filepath.c_str(), "r");

        if (!file) {
            return std::nullopt;
        }

        Parser::Data data;

        char name[64];
        char edgeWeightType[64];

        if (fscanf(file, "NAME : %s\n", name) != 1) return std::nullopt;
        if (fscanf(file, "COMMENT : %*[^\n]\n") != 0) return std::nullopt;
        if (fscanf(file, "TYPE : %*[^\n]\n") != 0) return std::nullopt;

        int matrix_size;
        if (fscanf(file, "DIMENSION : %d\n", &matrix_size) != 1) return std::nullopt;
        if (fscanf(file, "EDGE_WEIGHT_TYPE : %s\n", edgeWeightType) != 1) return std::nullopt;

        if (fscanf(file, "CAPACITY : %d\n", &data.vehicle_capacity) != 1) return std::nullopt;

        data.xcoords.resize(matrix_size);
        data.ycoords.resize(matrix_size);

        if (fscanf(file, "NODE_COORD_SECTION\n") != 0) return std::nullopt;

        int vertex_index;
        for (int i = 0; i < matrix_size; ++i) {
            if (fscanf(file, "%d %lf %lf", &vertex_index, &data.xcoords[i], &data.ycoords[i]) != 3) return std::nullopt;
        }

        if (fscanf(file, "\n") != 0) return std::nullopt;
        if (fscanf(file, "DEMAND_SECTION\n") != 0) return std::nullopt;

        data.demands.resize(matrix_size);
        for (int i = 0; i < matrix_size; i++) {
            if (fscanf(file, "%d %d", &vertex_index, &data.demands[i]) != 2) return std::nullopt;
        }

        fclose(file);

        return data;
    }

}  // namespace cobra