#ifndef _FILO2_PARAMETERS_HPP_
#define _FILO2_PARAMETERS_HPP_

#include <filesystem>
#include <iostream>
#include <string>
#include <optional> // 确保包含了 <optional>

// Default parameters.
#define DEFAULT_OUTPATH ("./")
#define DEFAULT_SOLUTION_CACHE_HISTORY (50)
#define DEFAULT_CW_LAMBDA (1.0)
#define DEFAULT_NEIGHBORS_NUM (1500)
#define DEFAULT_CW_NEIGHBORS (100)
#define DEFAULT_ROUTEMIN_ITERATIONS (1000)
#define DEFAULT_COREOPT_ITERATIONS (100000)
#define DEFAULT_SPARSIFICATION_RULE1_NEIGHBORS (25)
#define DEFAULT_SPARSIFICATION_FACTOR (0.25)
#define DEFAULT_SPARSIFICATION_MULTIPLIER (0.50)
#define DEFAULT_SHAKING_LB_FACTOR (0.375)
#define DEFAULT_SHAKING_UB_FACTOR (0.85)
#define DEFAULT_TOLERANCE (0.01)
#define DEFAULT_SEED (0)
#define DEFAULT_SA_INIT_FACTOR (0.1)
#define DEFAULT_SA_FINAL_FACTOR (0.01)

#define DEFAULT_USE_ROUTEMIN (true)

// Tokens.
#define TOKEN_OUTPATH ("--outpath")
#define TOKEN_TOLERANCE ("--tolerance")
#define TOKEN_NEIGHBORS_NUM ("--neighbors-num")
#define TOKEN_SPARSIFICATION_RULE1_NEIGHBORS ("--granular-neighbors")
#define TOKEN_SOLUTION_CACHE_HISTORY ("--cache")
#define TOKEN_ROUTEMIN_ITERATIONS ("--routemin-iterations")
#define TOKEN_COREOPT_ITERATIONS ("--coreopt-iterations")
#define TOKEN_SPARSIFICATION_FACTOR ("--granular-gamma-base")
#define TOKEN_SPARSIFICATION_MULTIPLIER ("--granular-delta")
#define TOKEN_SHAKING_LB_FACTOR ("--shaking-lower-bound")
#define TOKEN_SHAKING_UB_FACTOR ("--shaking-upper-bound")
#define TOKEN_SEED ("--seed")
#define TOKEN_HELP ("--help")
#define TOKEN_SA_INIT_FACTOR ("--sa-initial-factor")
#define TOKEN_SA_FINAL_FACTOR ("--sa-final-factor")

#define TOKEN_USE_ROUTEMIN ("--use-routemin")


// == 在这里添加下面两行 ==
#define TOKEN_TIME ("-t")
#define TOKEN_TIME_LONG ("--time")
// == 添加结束 ==

class Parameters {

public:
    explicit Parameters(int argc, char* argv[]) {

        if (argc == 1) {
            std::cout << "Missing input instance.\n\n";
            exit(EXIT_FAILURE);
        }

        instance_path = std::string(argv[1]);

        for (auto n = 2; n < argc; n += 2) {

            auto token = std::string(argv[n]);

            if (n + 1 >= argc) {
                std::cout << "Missing value for '" << token << "'.\n\n";
                exit(EXIT_FAILURE);
            }
            auto value = std::string(argv[n + 1]);

            set(token, value);
        }
    }

    // 在 Parameters.hpp 的 public: 部分
    auto get_max_execution_seconds() const -> std::optional<long> {
        return max_execution_seconds;
    }

    inline int get_solution_cache_size() const {
        return solution_cache_history;
    }
    inline double get_cw_lambda() const {
        return cw_lambda;
    }
    inline int get_cw_neighbors() const {
        return cw_neighbors;
    }
    inline int get_routemin_iterations() const {
        return routemin_iterations;
    }
    inline int get_coreopt_iterations() const {
        return coreopt_iterations;
    }
    inline int get_sparsification_rule_neighbors() const {
        return sparsification_rule_neighbors;
    }
    inline double get_gamma_base() const {
        return gamma_base;
    }
    inline double get_delta() const {
        return delta;
    }
    inline double get_shaking_lb_factor() const {
        return shaking_lb_factor;
    }
    inline double get_shaking_ub_factor() const {
        return shaking_ub_factor;
    }
    inline double get_tolerance() const {
        return tolerance;
    }
    inline std::string get_instance_path() const {
        return instance_path;
    }
    inline std::string get_outpath() const {
        return outpath;
    }
    inline int get_seed() const {
        return seed;
    }
    inline auto get_sa_initial_factor() const {
        return sa_initial_factor;
    }
    inline auto get_sa_final_factor() const {
        return sa_final_factor;
    }

    inline int get_neighbors_num() const {
        return neighbors_num;
    }

    inline bool get_use_routemin() const {
        return use_routemin;
    }

    void set(const std::string& key, const std::string& value) {

        if (key == TOKEN_OUTPATH) {
            outpath = value;
            if (outpath.back() != std::filesystem::path::preferred_separator) {
                outpath += std::filesystem::path::preferred_separator;
            }
        } else if (key == TOKEN_TOLERANCE) {
            tolerance = std::stof(value);
        } else if (key == TOKEN_SPARSIFICATION_RULE1_NEIGHBORS) {
            sparsification_rule_neighbors = std::stoi(value);
        } else if (key == TOKEN_SOLUTION_CACHE_HISTORY) {
            solution_cache_history = std::stoi(value);
        } else if (key == TOKEN_ROUTEMIN_ITERATIONS) {
            routemin_iterations = std::stoi(value);
        } else if (key == TOKEN_COREOPT_ITERATIONS) {
            coreopt_iterations = std::stoi(value);
        } else if (key == TOKEN_SPARSIFICATION_FACTOR) {
            gamma_base = std::stof(value);
        } else if (key == TOKEN_SPARSIFICATION_MULTIPLIER) {
            delta = std::stof(value);
        } else if (key == TOKEN_SHAKING_LB_FACTOR) {
            shaking_lb_factor = std::stof(value);
        } else if (key == TOKEN_SHAKING_UB_FACTOR) {
            shaking_ub_factor = std::stof(value);
        } else if (key == TOKEN_SEED) {
            seed = std::stoi(value);
        } else if (key == TOKEN_SA_INIT_FACTOR) {
            sa_initial_factor = std::stof(value);
        } else if (key == TOKEN_SA_FINAL_FACTOR) {
            sa_final_factor = std::stof(value);
        } else if (key == TOKEN_NEIGHBORS_NUM) {
            neighbors_num = std::stoi(value);
        } else if (key == TOKEN_TIME || key == TOKEN_TIME_LONG) {
            max_execution_seconds = std::stol(value); // 使用 stol (string to long)
        } else if (key == TOKEN_USE_ROUTEMIN) {
            if (value == "false") {
                use_routemin = false;
            } else {
                use_routemin = true;
            }
        } else {
            std::cout << "Error: unknown argument '" << key << "'. Try --help for more information.\n";
            exit(EXIT_SUCCESS);
        }
    }

private:
    std::string instance_path;
    std::string outpath = DEFAULT_OUTPATH;
    double tolerance = DEFAULT_TOLERANCE;
    int solution_cache_history = DEFAULT_SOLUTION_CACHE_HISTORY;
    double cw_lambda = DEFAULT_CW_LAMBDA;
    int cw_neighbors = DEFAULT_CW_NEIGHBORS;
    int routemin_iterations = DEFAULT_ROUTEMIN_ITERATIONS;
    int coreopt_iterations = DEFAULT_COREOPT_ITERATIONS;
    int sparsification_rule_neighbors = DEFAULT_SPARSIFICATION_RULE1_NEIGHBORS;
    double gamma_base = DEFAULT_SPARSIFICATION_FACTOR;
    double delta = DEFAULT_SPARSIFICATION_MULTIPLIER;
    double shaking_lb_factor = DEFAULT_SHAKING_LB_FACTOR;
    double shaking_ub_factor = DEFAULT_SHAKING_UB_FACTOR;
    int seed = DEFAULT_SEED;
    double sa_initial_factor = DEFAULT_SA_INIT_FACTOR;
    double sa_final_factor = DEFAULT_SA_FINAL_FACTOR;
    int neighbors_num = DEFAULT_NEIGHBORS_NUM;
    std::optional<long> max_execution_seconds;
    bool use_routemin = DEFAULT_USE_ROUTEMIN;
};


#endif