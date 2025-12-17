#include <fstream>

#include "Parameters.hpp"
#include "base/PrettyPrinter.hpp"
#include "base/Timer.hpp"
#include "base/Welford.hpp"
#include "instance/Instance.hpp"
#include "localsearch/LocalSearch.hpp"
#include "movegen/MoveGenerators.hpp"
#include "opt/RuinAndRecreate.hpp"
#include "opt/SimulatedAnnealing.hpp"
#include "opt/bpp.hpp"
#include "opt/routemin.hpp"
#include "solution/Solution.hpp"
#include "solution/savings.hpp"
#include <filesystem>       // NOTE: add this line
#include <iomanip>

#ifdef GUI
    #include "Renderer.hpp"
#endif

auto get_basename(const std::string& pathname) -> std::string {
    return {std::find_if(pathname.rbegin(), pathname.rend(), [](char c) { return c == '/'; }).base(), pathname.end()};
}


// Few notes:
// - Inputs are never checked when the solver is compiled in release mode. There are just a lot of assertions checked in debug mode.
int main(int argc, char* argv[]) {


#ifndef NDEBUG
    std::cout << "******************************\n";
    std::cout << "Probably running in DEBUG mode\n";
    std::cout << "******************************\n\n";
#endif

    cobra::Timer global_timer;
#ifdef VERBOSE
    cobra::Timer timer;
#endif

    const auto params = Parameters(argc, argv);

#ifdef VERBOSE
    std::cout << "Pre-processing the instance.\n";
    timer.reset();
#endif
    std::optional<cobra::Instance> maybe_instance = cobra::Instance::make(params.get_instance_path(), params.get_neighbors_num());
#ifdef VERBOSE
    std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n\n";
#endif

    if (!maybe_instance.has_value()) {
        return EXIT_FAILURE;
    }

    const cobra::Instance instance = std::move(maybe_instance.value());

    auto best_solution = cobra::Solution(instance, std::min(instance.get_vertices_num(), params.get_solution_cache_size()));

#ifdef VERBOSE
    std::cout << "Running CLARKE&WRIGHT to generate an initial solution.\n";
    timer.reset();
#endif
    cobra::clarke_and_wright(instance, best_solution, params.get_cw_lambda(), params.get_cw_neighbors());
#ifdef VERBOSE
    std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n";
    std::cout << "Initial solution: obj = " << best_solution.get_cost() << ", n. of routes = " << best_solution.get_routes_num() << ".\n\n";
#endif

    auto k = params.get_sparsification_rule_neighbors();

#ifdef VERBOSE
    std::cout << "Setting up MOVEGENERATORS data structures.\n";
    timer.reset();
#endif

    auto move_generators = cobra::MoveGenerators(instance, k);

#ifdef VERBOSE
    std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n";
    const auto tot_arcs = static_cast<unsigned long>(instance.get_vertices_num()) * static_cast<unsigned long>(instance.get_vertices_num());
    const auto move_gen_num = move_generators.size();
    const auto move_gen_perc = 100.0 * static_cast<double>(move_gen_num) / static_cast<double>(tot_arcs);
    std::cout << "Using at most " << move_generators.size() << " move-generators out of " << tot_arcs << " total arcs ";
    std::cout << std::fixed;
    std::cout << std::setprecision(5);
    std::cout << "(approx. " << move_gen_perc << "%)\n\n";
    std::cout << std::defaultfloat;
#endif


#ifdef VERBOSE
    std::cout << "Computing a greedy upper bound on the n. of routes.\n";
    timer.reset();
#endif

    auto kmin = bpp::greedy_first_fit_decreasing(instance);

#ifdef VERBOSE
    std::cout << "Done in " << timer.elapsed_time<std::chrono::milliseconds>() << " milliseconds.\n";
    std::cout << "Around " << kmin << " routes should do the job.\n\n";
#endif

    auto rand_engine = std::mt19937(params.get_seed());
    const auto tolerance = params.get_tolerance();

    if (kmin < best_solution.get_routes_num()) {

        const auto routemin_iterations = params.get_routemin_iterations();

#ifdef VERBOSE
        std::cout << "Running ROUTEMIN heuristic for at most " << routemin_iterations << " iterations.\n";
        std::cout << "Starting solution: obj = " << best_solution.get_cost() << ", n. of routes = " << best_solution.get_routes_num()
                  << ".\n";
        timer.reset();
#endif
        // Note: route minimization
        if (params.get_use_routemin()) {
            best_solution = routemin(instance, best_solution, rand_engine, move_generators, kmin, routemin_iterations, tolerance);
        }
//        best_solution = routemin(instance, best_solution, rand_engine, move_generators, kmin, routemin_iterations, tolerance);

#ifdef VERBOSE
        std::cout << "Final solution: obj = " << best_solution.get_cost() << ", n. routes = " << best_solution.get_routes_num() << "\n";
        std::cout << "Done in " << timer.elapsed_time<std::chrono::seconds>() << " seconds.\n\n";
#endif
    }


    auto rvnd0 = cobra::RandomizedVariableNeighborhoodDescent(
        instance, move_generators,
        {cobra::E11,   cobra::E10,   cobra::TAILS, cobra::SPLIT, cobra::RE22B, cobra::E22,  cobra::RE20,  cobra::RE21,
         cobra::RE22S, cobra::E21,   cobra::E20,   cobra::TWOPT, cobra::RE30,  cobra::E30,  cobra::RE33B, cobra::E33,
         cobra::RE31,  cobra::RE32B, cobra::RE33S, cobra::E31,   cobra::E32,   cobra::RE32S},
        rand_engine, tolerance);

    auto rvnd1 = cobra::RandomizedVariableNeighborhoodDescent(instance, move_generators, {cobra::EJCH}, rand_engine, tolerance);

    auto local_search = cobra::VariableNeighborhoodDescentComposer(tolerance);
    local_search.append(&rvnd0);
    local_search.append(&rvnd1);


    const auto coreopt_iterations = params.get_coreopt_iterations();
    const auto max_seconds = params.get_max_execution_seconds();

    auto neighbor = best_solution;

    const auto gamma_base = params.get_gamma_base();
    auto gamma = std::vector<double>(instance.get_vertices_num(), gamma_base);
    auto gamma_counter = std::vector<int>(instance.get_vertices_num(), 0);

    const auto delta = params.get_delta();
    auto average_number_of_vertices_accessed = cobra::Welford();

    auto gamma_vertices = std::vector<int>();
    for (auto i = instance.get_vertices_begin(); i < instance.get_vertices_end(); i++) {
        gamma_vertices.emplace_back(i);
    }
    move_generators.set_active_percentage(gamma, gamma_vertices);

    auto ruined_customers = std::vector<int>();

    auto rr = RuinAndRecreate(instance, rand_engine);

    const auto intensification_lb = params.get_shaking_lb_factor();
    const auto intensification_ub = params.get_shaking_ub_factor();

    const auto mean_solution_arc_cost = neighbor.get_cost() / (static_cast<double>(instance.get_customers_num()) +
                                                               2.0 * static_cast<double>(neighbor.get_routes_num()));

    auto shaking_lb_factor = mean_solution_arc_cost * intensification_lb;
    auto shaking_ub_factor = mean_solution_arc_cost * intensification_ub;

#ifdef VERBOSE
    std::cout << "Shaking LB = " << shaking_lb_factor << "\n";
    std::cout << "Shaking UB = " << shaking_ub_factor << "\n";
#endif

    const auto omega_base = std::max(1, static_cast<int>(std::ceil(std::log(instance.get_vertices_num()))));
    auto omega = std::vector<int>(instance.get_vertices_num(), omega_base);
    auto random_choice = std::uniform_int_distribution(0, 1);

    auto vertices_dist = std::uniform_int_distribution(instance.get_vertices_begin(), instance.get_vertices_end() - 1);
    cobra::Welford welf;
    for (int i = 0; i < instance.get_vertices_num(); ++i) {
        welf.update(instance.get_cost(vertices_dist(rand_engine), vertices_dist(rand_engine)));
    }

    const auto sa_initial_temperature = welf.get_mean() * params.get_sa_initial_factor();
    const auto sa_final_temperature = sa_initial_temperature * params.get_sa_final_factor();

    auto sa = cobra::SimulatedAnnealing(sa_initial_temperature, sa_final_temperature, rand_engine, coreopt_iterations);

#ifdef VERBOSE
    std::cout << "Simulated annealing temperature goes from " << sa_initial_temperature << " to " << sa_final_temperature << ".\n\n";
#endif


#ifdef VERBOSE
    if (max_seconds.has_value()) {
        std::cout << "Running COREOPT for " << max_seconds.value() << " seconds.\n";
    } else {
        std::cout << "Running COREOPT for " << coreopt_iterations << " iterations.\n";
    }

    auto welford_rac_before_shaking = cobra::Welford();
    auto welford_rac_after_shaking = cobra::Welford();
    auto welford_local_optima = cobra::Welford();
    auto welford_shaken_solutions = cobra::Welford();
    auto printer = cobra::PrettyPrinter({{"%", cobra::PrettyPrinter::Field::Type::REAL, 5, " "},
                                         {"Iterations", cobra::PrettyPrinter::Field::Type::INTEGER, 10, " "},
                                         {"Objective", cobra::PrettyPrinter::Field::Type::INTEGER, 10, " "},
                                         {"Routes", cobra::PrettyPrinter::Field::Type::INTEGER, 6, " "},
                                         {"Iter/s", cobra::PrettyPrinter::Field::Type::REAL, 10, " "},
                                         {"Eta (s)", cobra::PrettyPrinter::Field::Type::REAL, 10, " "},
                                         {"RR (micro)", cobra::PrettyPrinter::Field::Type::REAL, 10, " "},
                                         {"LS (micro)", cobra::PrettyPrinter::Field::Type::REAL, 10, " "},
                                         {"Gamma", cobra::PrettyPrinter::Field::Type::REAL, 5, " "},
                                         {"Omega", cobra::PrettyPrinter::Field::Type::REAL, 6, " "},
                                         {"Temp", cobra::PrettyPrinter::Field::Type::REAL, 6, " "}});

    auto elapsed_minutes = 0UL;
    timer.reset();
    cobra::Timer coreopt_timer;

    cobra::Welford welford_rr;
    cobra::Welford welford_ls;

#endif

    // todo
    // ==========================================================
    // == 在这里添加代码 (第 1 部分) ==

    // 1. 构造 CSV 文件名
    const auto progress_csv_file = params.get_outpath() + get_basename(params.get_instance_path()) + "_seed-" +
                                   std::to_string(params.get_seed()) + ".vrp.csv";

    // 2. 确保目录存在
    std::filesystem::create_directories(params.get_outpath());

    // 3. 以追加模式 (std::ios_base::app) 打开文件
    std::ofstream progress_stream(progress_csv_file, std::ios_base::app);
    progress_stream << std::setprecision(10); // 设置精度

    // 4. 记录初始解 (相对时间为 0.0)
    progress_stream << 0.0 << ";" << best_solution.get_cost() << "\n" << std::flush;

    // 5. 启动算法迭代循环的计时器
    cobra::Timer algorithm_timer;
    std::clock_t cpu_start_time = std::clock();

    // == 添加代码结束 ==
    // ==========================================================


#ifdef GUI
    auto renderer = Renderer(instance, neighbor.get_cost());
#endif

    // Cost of the working solution, from which neighbor is obtained after shaking and local search.
    double reference_solution_cost = neighbor.get_cost();

    for (long iter = 0; ; iter++) { // 无限循环，'long' 以防迭代次数过多

        // ==========================================
        // == 在这里添加终止检查 ==
        if (max_seconds.has_value()) {
//            // 基于墙上时间的终止
//            if (algorithm_timer.elapsed_time<std::chrono::seconds>() >= max_seconds.value()) {
//                break; // 退出循环
//            }
            // 基于CPU时间的终止
            double cpu_time_used = static_cast<double>(std::clock() - cpu_start_time) / CLOCKS_PER_SEC;
            if (cpu_time_used >= max_seconds.value()) {
                break; // 退出循环
            }

        } else {
            // 基于迭代的终止
            if (iter >= coreopt_iterations) {
                break; // 退出循环
            }
        }
        // == 终止检查结束 ==
        // ==========================================

//    for (auto iter = 0; iter < coreopt_iterations; iter++) {

        neighbor.apply_undo_list1(neighbor);
        neighbor.clear_do_list1();
        neighbor.clear_undo_list1();
        neighbor.clear_svc();

#ifdef VERBOSE
        if (global_timer.elapsed_time<std::chrono::minutes>() >= elapsed_minutes + 5) {
            printer.notify("Optimizing for " + std::to_string(global_timer.elapsed_time<std::chrono::minutes>()) + " minutes.");
            elapsed_minutes += 5;
        }

        cobra::Timer rr_timer;
#endif

        const auto walk_seed = rr.apply(neighbor, omega);


#ifdef VERBOSE
        const auto rr_time = rr_timer.elapsed_time<std::chrono::microseconds>();
        welford_rr.update(rr_time);
#endif

#ifdef GUI
        const auto shaken_solution_cost = neighbor.get_cost();
#endif

        ruined_customers.clear();
        for (auto i = neighbor.get_svc_begin(); i != neighbor.get_svc_end(); i = neighbor.get_svc_next(i)) {
            ruined_customers.emplace_back(i);
        }

#ifdef VERBOSE
        welford_rac_after_shaking.update(static_cast<double>(neighbor.get_svc_size()));
        welford_shaken_solutions.update(neighbor.get_cost());

        cobra::Timer ls_timer;
#endif

        local_search.sequential_apply(neighbor);

#ifdef VERBOSE
        const auto ls_time = ls_timer.elapsed_time<std::chrono::microseconds>();
        welford_ls.update(ls_time);
#endif

#ifdef GUI

        const auto local_optimum_cost = neighbor.get_cost();
#endif

        average_number_of_vertices_accessed.update(static_cast<double>(neighbor.get_svc_size()));

        auto max_non_improving_iterations = static_cast<int>(std::ceil(delta * static_cast<double>(coreopt_iterations) *
                                                                       static_cast<double>(average_number_of_vertices_accessed.get_mean()) /
                                                                       static_cast<double>(instance.get_vertices_num())));

#ifdef GUI
        if (iter % 1000 == 0) {
            renderer.draw(best_solution, neighbor.get_svc(), move_generators);
        }
#endif

#ifdef VERBOSE
        welford_rac_before_shaking.update(static_cast<double>(neighbor.get_svc_size()));
        welford_local_optima.update(neighbor.get_cost());
#endif

        bool improved_best_solution;

        if (neighbor.get_cost() < best_solution.get_cost()) {

            // best_solution = solution;

            improved_best_solution = true;

            // TODO: 添加输出csv记录
            // ==========================================================
            // == 在这里添加代码 (第 2 部分) ==

            // 1. 获取相对于 algorithm_timer 的时间（浮点数秒）
            const auto relative_sec = algorithm_timer.elapsed_time<std::chrono::microseconds>() / 1'000'000.0;

            // 2. 获取新成本
            const auto new_cost = neighbor.get_cost();

            // 3. 按照格式 {time};{cost}\n 写入并刷新
            progress_stream << relative_sec << ";" << new_cost << "\n" << std::flush;

            // == 添加代码结束 ==
            // ==========================================================

            neighbor.apply_do_list2(best_solution);
            neighbor.apply_do_list1(best_solution);  // latest changes
            neighbor.clear_do_list2();

            assert(best_solution == neighbor);


            gamma_vertices.clear();
            for (auto i = neighbor.get_svc_begin(); i != neighbor.get_svc_end(); i = neighbor.get_svc_next(i)) {
                gamma[i] = gamma_base;
                gamma_counter[i] = 0;
                gamma_vertices.emplace_back(i);
            }
            move_generators.set_active_percentage(gamma, gamma_vertices);

#ifdef VERBOSE
            welford_local_optima.reset();
            welford_local_optima.update(neighbor.get_cost());
            welford_shaken_solutions.reset();
            welford_shaken_solutions.update(neighbor.get_cost());
#endif

        } else {

            improved_best_solution = false;

            for (auto i = neighbor.get_svc_begin(); i != neighbor.get_svc_end(); i = neighbor.get_svc_next(i)) {
                gamma_counter[i]++;
                if (gamma_counter[i] >= max_non_improving_iterations) {
                    gamma[i] = std::min(gamma[i] * 2.0, 1.0);
                    gamma_counter[i] = 0;
                    gamma_vertices.clear();
                    gamma_vertices.emplace_back(i);
                    move_generators.set_active_percentage(gamma, gamma_vertices);
                }
            }
        }

        const auto seed_shake_value = omega[walk_seed];

        if (neighbor.get_cost() > shaking_ub_factor + reference_solution_cost) {
            for (auto i : ruined_customers) {
                if (omega[i] > seed_shake_value - 1) {
                    omega[i]--;
                }
            }
        } else if (neighbor.get_cost() >= reference_solution_cost && neighbor.get_cost() < reference_solution_cost + shaking_lb_factor) {
            for (auto i : ruined_customers) {
                if (omega[i] < seed_shake_value + 1) {
                    omega[i]++;
                }
            }
        } else {
            for (auto i : ruined_customers) {
                if (random_choice(rand_engine)) {
                    if (omega[i] > seed_shake_value - 1) {
                        omega[i]--;
                    }
                } else {
                    if (omega[i] < seed_shake_value + 1) {
                        omega[i]++;
                    }
                }
            }
        }

        if (sa.accept(reference_solution_cost, neighbor)) {

            if (!improved_best_solution) {
                neighbor.append_do_list1_to_do_list2();
            }

            neighbor.clear_do_list1();
            neighbor.clear_undo_list1();

            reference_solution_cost = neighbor.get_cost();

            const auto updated_mean_solution_arc_cost = neighbor.get_cost() / (static_cast<double>(instance.get_customers_num()) +
                                                                               2.0 * static_cast<double>(neighbor.get_routes_num()));
            shaking_lb_factor = updated_mean_solution_arc_cost * intensification_lb;
            shaking_ub_factor = updated_mean_solution_arc_cost * intensification_ub;
        }

        sa.decrease_temperature();

#ifdef GUI
        renderer.add_trajectory_point(shaken_solution_cost, local_optimum_cost, reference_solution_cost, best_solution.get_cost());
#endif

#ifdef VERBOSE
        if (timer.elapsed_time<std::chrono::seconds>() > 1) {
            timer.reset();

            const auto elapsed_seconds = algorithm_timer.elapsed_time<std::chrono::seconds>();
            const auto iter_per_second = static_cast<double>(iter + 1) / (static_cast<double>(elapsed_seconds) + 0.01);

            double progress = 0.0;
            double estimated_rem_time = 0.0;

            if (max_seconds.has_value()) {
                // 基于时间的进度
                progress = 100.0 * static_cast<double>(elapsed_seconds) / static_cast<double>(max_seconds.value());
                estimated_rem_time = static_cast<double>(max_seconds.value() - elapsed_seconds);
            } else {
                // 基于迭代的进度
                progress = 100.0 * (iter + 1.0) / coreopt_iterations;
                const auto remaining_iter = coreopt_iterations - (iter + 1); // iter 从0开始，所以用 iter+1
                estimated_rem_time = static_cast<double>(remaining_iter) / iter_per_second;
            }

//            const auto progress = 100.0 * (iter + 1.0) / coreopt_iterations;
//            const auto elapsed_seconds = coreopt_timer.elapsed_time<std::chrono::seconds>();
//            const auto iter_per_second = static_cast<double>(iter + 1) / (static_cast<double>(elapsed_seconds) + 0.01);
//            const auto remaining_iter = coreopt_iterations - iter;
//            const auto estimated_rem_time = static_cast<double>(remaining_iter) / iter_per_second;

            auto gamma_mean = 0.0;
            for (auto i = instance.get_vertices_begin(); i < instance.get_vertices_end(); i++) {
                gamma_mean += gamma[i];
            }
            gamma_mean = (gamma_mean / static_cast<double>(instance.get_vertices_num()));

            auto omega_mean = 0.0;
            for (auto i = instance.get_customers_begin(); i < instance.get_customers_end(); i++) {
                omega_mean += omega[i];
            }
            omega_mean /= static_cast<double>(instance.get_customers_num());


            printer.print(progress, iter + 1, best_solution.get_cost(), best_solution.get_routes_num(), iter_per_second, estimated_rem_time,
                          welford_rr.get_mean(), welford_ls.get_mean(), gamma_mean, omega_mean, sa.get_temperature());
        }
#endif
    }

    // TODO
    // ==========================================================
    // == 在这里添加代码（第 3 部分） ==
    progress_stream.close(); // 关闭 CSV 文件流
    // == 添加代码结束 ==
    // ==========================================================

    int global_time_elapsed = global_timer.elapsed_time<std::chrono::seconds>();

#ifdef VERBOSE
    std::cout << "\n";
    std::cout << "Best solution found:\n";
    std::cout << "obj = " << best_solution.get_cost() << ", n. routes = " << best_solution.get_routes_num() << "\n";

    std::cout << "\n";
    std::cout << "Run completed in " << global_time_elapsed << " seconds ";
#endif

    const auto outfile = params.get_outpath() + get_basename(params.get_instance_path()) + "_seed-" + std::to_string(params.get_seed()) +
                         ".out";

    std::filesystem::create_directories(params.get_outpath());

    auto out_stream = std::ofstream(outfile);
    out_stream << std::setprecision(10);
    out_stream << best_solution.get_cost() << "\t" << global_time_elapsed << "\n";
    cobra::Solution::store_to_file(
        instance, best_solution,
        params.get_outpath() + get_basename(params.get_instance_path()) + "_seed-" + std::to_string(params.get_seed()) + ".vrp.sol");

#ifdef VERBOSE
    std::cout << "\n";
    std::cout << "Results stored in\n";
    std::cout << " - " << outfile << "\n";
    std::cout << " - "
              << params.get_outpath() + get_basename(params.get_instance_path()) + "_seed-" + std::to_string(params.get_seed()) + ".vrp.sol"
              << "\n";
#endif

    return EXIT_SUCCESS;
}