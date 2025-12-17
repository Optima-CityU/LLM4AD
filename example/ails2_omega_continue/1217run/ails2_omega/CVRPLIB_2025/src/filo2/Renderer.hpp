#ifndef _FILO2_RENDERER_HPP_
#define _FILO2_RENDERER_HPP_

#include <GLFW/glfw3.h>

#include "instance/Instance.hpp"
#include "movegen/MoveGenerators.hpp"
#include "solution/Solution.hpp"

#define FRAME_WIDTH (10)
#define TRAJECTORY_SECTION_WIDTH (300)
#define SOLUTION_SECTION_WIDTH (700)

#define TRAJECTORY_SECTION_BEGIN (FRAME_WIDTH)
#define TRAJECTORY_SECTION_END (TRAJECTORY_SECTION_BEGIN + TRAJECTORY_SECTION_WIDTH)

#define SOLUTION_SECTION_BEGIN (TRAJECTORY_SECTION_END + FRAME_WIDTH)
#define SOLUTION_SECTION_END (SOLUTION_SECTION_BEGIN + SOLUTION_SECTION_WIDTH)

#define YMIN (0)
#define YMAX (SOLUTION_SECTION_END + FRAME_WIDTH)

#define XMIN (0)
#define XMAX (700)

class Renderer {

public:
    Renderer(const cobra::Instance& instance_, double initial_cost_) : instance(instance_), initial_cost(initial_cost_) {

        if (!glfwInit()) {
            std::cerr << "Cannot successfully execute 'glfwInit'\n";
            abort();
        }

        window = glfwCreateWindow(900, 900, "FILO", nullptr, nullptr);

        if (!window) {
            std::cerr << "Cannot successfully execute 'glfwCreateWindow'\n";
            abort();
        }

        glfwSetWindowCloseCallback(window, [](GLFWwindow*) -> void { exit(1); });

        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* wind, int width, int height) {
            glfwMakeContextCurrent(wind);
            glViewport(0, 0, width, height);
        });

        glfwMakeContextCurrent(window);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        glEnable(GL_LINE_SMOOTH);

        glfwPollEvents();

        x_min = instance.get_x_coordinate(instance.get_vertices_begin());
        x_max = x_min;
        y_min = instance.get_y_coordinate(instance.get_vertices_begin());
        y_max = y_min;
        for (auto i = instance.get_vertices_begin(); i < instance.get_vertices_end(); i++) {
            const auto x_i = instance.get_x_coordinate(i);
            const auto y_i = instance.get_y_coordinate(i);
            x_min = std::min(x_min, x_i);
            x_max = std::max(x_max, x_i);
            y_min = std::min(y_min, y_i);
            y_max = std::max(y_max, y_i);
        }

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        glEnable(GL_LINE_SMOOTH);

        add_trajectory_point(initial_cost, initial_cost, initial_cost, initial_cost);
    }

    void add_trajectory_point(double shaken_solution_cost, double local_optimum_cost, double current_solution_cost,
                              double best_solution_cost) {

        auto trajectory_point = TrajectoryPoint();

        trajectory_point.current_solution_gap = 100.0 * (current_solution_cost - initial_cost) / initial_cost;
        trajectory_point.best_solution_gap = 100.0 * (best_solution_cost - initial_cost) / initial_cost;
        trajectory_point.shaken_solution_gap = 100.0 * (shaken_solution_cost - initial_cost) / initial_cost;
        trajectory_point.local_optima_gap = 100.0 * (local_optimum_cost - initial_cost) / initial_cost;
        trajectory.emplace_back(trajectory_point);

        min_gap = std::min(trajectory_point.best_solution_gap, min_gap);

        max_gap = std::max(trajectory_point.local_optima_gap, max_gap);
    }

    void draw(cobra::Solution& solution, const cobra::LRUCache& cached_vertices, cobra::MoveGenerators& move_generators) {

        glfwMakeContextCurrent(window);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();

        glOrtho(XMIN, XMAX, YMIN, YMAX, -1, 1);

        glMatrixMode(GL_MODELVIEW);

        glClearColor(0.14, 0.15, 0.16, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        draw_trajectory();

        draw_hseparator(YMIN, FRAME_WIDTH);
        draw_hseparator(TRAJECTORY_SECTION_END, SOLUTION_SECTION_BEGIN);

        draw_solution(solution, cached_vertices, move_generators);

        draw_hseparator(SOLUTION_SECTION_END, YMAX);

        draw_vseparator(XMIN, XMIN + FRAME_WIDTH);
        draw_vseparator(XMAX - FRAME_WIDTH, XMAX);

        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    static void draw_hseparator(double begin, double end) {

        glBegin(GL_QUADS);
        glColor4f(0.18, 0.19, 0.20, 1);
        glVertex2d(XMIN, begin);
        glVertex2d(XMAX, begin);
        glVertex2d(XMAX, end);
        glVertex2d(XMIN, end);
        glEnd();

        glColor4f(0.25, 0.26, 0.27, 1);
        glLineWidth(1);

        glBegin(GL_LINES);
        glVertex2d(XMIN, begin);
        glVertex2d(XMAX, begin);
        glEnd();

        glBegin(GL_LINES);
        glVertex2d(XMIN, end);
        glVertex2d(XMAX, end);
        glEnd();
    }

    static void draw_vseparator(double begin, double end) {

        glBegin(GL_QUADS);
        glColor4f(0.18, 0.19, 0.20, 1);
        glVertex2d(begin, YMIN);
        glVertex2d(end, YMIN);
        glVertex2d(end, YMAX);
        glVertex2d(begin, YMAX);
        glEnd();

        glColor4f(0.25, 0.26, 0.27, 1);
        glLineWidth(1);

        glBegin(GL_LINES);
        glVertex2d(begin, YMIN);
        glVertex2d(begin, YMAX);
        glEnd();

        glBegin(GL_LINES);
        glVertex2d(end, YMIN);
        glVertex2d(end, YMAX);
        glEnd();
    }

    void draw_trajectory() {

        auto xs = Scaler(0, trajectory.size(), XMIN + FRAME_WIDTH, XMAX - FRAME_WIDTH);
        auto ys = Scaler(min_gap - 0.5, max_gap + 1, TRAJECTORY_SECTION_BEGIN, TRAJECTORY_SECTION_END);

        glColor4f(1.0, 1.0, 1.0, 0.25);
        glPushAttrib(GL_ENABLE_BIT);
        glLineStipple(5, 0xAAAA);
        glEnable(GL_LINE_STIPPLE);
        glBegin(GL_LINES);
        for (auto i = std::min(static_cast<int>(std::round(min_gap - 0.5)), 1); i <= max_gap + 1; i++) {
            glVertex2f(xs.scale(0), ys.scale(i));
            glVertex2f(xs.scale(trajectory.size()), ys.scale(i));
        }
        glEnd();
        glPopAttrib();


        glColor4f(1.0, 0.0, 0.0, 0.50);
        glBegin(GL_POINTS);
        for (auto x = 0u; x < trajectory.size(); x++) {
            const auto y = trajectory[x].shaken_solution_gap;
            glVertex2f(xs.scale(x), ys.scale(y));
        }
        glEnd();


        glColor4f(0.0, 1.0, 0.0, 0.50);
        glBegin(GL_POINTS);
        for (auto x = 0u; x < trajectory.size(); x++) {
            const auto y = trajectory[x].local_optima_gap;
            glVertex2f(xs.scale(x), ys.scale(y));
        }
        glEnd();


        glBegin(GL_LINES);
        for (auto x = 0u; x < trajectory.size() - 1; x++) {
            glColor4f(1.0, 1.0, 0.0, 1.0);
            const auto y = trajectory[x].current_solution_gap;
            glVertex2f(xs.scale(x), ys.scale(y));
            glVertex2f(xs.scale(x + 1), ys.scale(trajectory[x + 1].current_solution_gap));
        }
        glEnd();


        glLineWidth(4.0);

        glColor4f(0.0, 0.0, 1.0, 1.0);
        glBegin(GL_LINES);
        for (auto x = 0u; x < trajectory.size() - 1; x++) {
            const auto y = trajectory[x].best_solution_gap;
            glVertex2f(xs.scale(x), ys.scale(y));
            glVertex2f(xs.scale(x + 1), ys.scale(trajectory[x + 1].best_solution_gap));
        }
        glEnd();
    }

    void draw_solution(cobra::Solution& solution, const cobra::LRUCache& cached_vertices, cobra::MoveGenerators& move_generators) {

        auto xs = Scaler(x_min, x_max, XMIN + FRAME_WIDTH, XMAX - FRAME_WIDTH);
        auto ys = Scaler(y_min, y_max, SOLUTION_SECTION_BEGIN, SOLUTION_SECTION_END);

        glLineWidth(1);
        for (auto i = instance.get_vertices_begin(); i < instance.get_vertices_end(); i++) {

            for (auto move_id : move_generators.get_move_generator_indices_involving(i)) {

                const auto& move = move_generators.get(move_id);

                glBegin(GL_LINES);
                glColor4f(0.0, 0.43, 0.75, 0.10);
                glVertex2f(xs.scale(instance.get_x_coordinate(move.get_first_vertex())),
                           ys.scale(instance.get_y_coordinate(move.get_first_vertex())));
                glVertex2f(xs.scale(instance.get_x_coordinate(move.get_second_vertex())),
                           ys.scale(instance.get_y_coordinate(move.get_second_vertex())));
                glEnd();
            }
        }


        for (auto i = cached_vertices.begin(); i != cached_vertices.end(); i = cached_vertices.get_next(i)) {

            for (auto move_id : move_generators.get_move_generator_indices_involving(i)) {
                const auto& move = move_generators.get(move_id);
                glBegin(GL_LINES);
                glColor4f(0.07, 0.60, 0.98, 0.20);
                glVertex2d(xs.scale(instance.get_x_coordinate(move.get_first_vertex())),
                           ys.scale(instance.get_y_coordinate(move.get_first_vertex())));
                glVertex2d(xs.scale(instance.get_x_coordinate(move.get_second_vertex())),
                           ys.scale(instance.get_y_coordinate(move.get_second_vertex())));
                glEnd();
            }
        }


        glLineWidth(2);
        for (auto route = solution.get_first_route(); route != cobra::Solution::dummy_route; route = solution.get_next_route(route)) {

            const auto load_ratio = static_cast<double>(solution.get_route_load(route)) /
                                    static_cast<double>(instance.get_vehicle_capacity());

            glBegin(GL_LINES);

            glColor4f(load_ratio, 1 - load_ratio, 0.00, 1.0);

            auto curr = solution.get_first_customer(route);
            auto next = curr;

            do {

                next = solution.get_next_vertex(route, curr);

                if (next == instance.get_depot()) {
                    break;
                }

                glVertex2f(xs.scale(instance.get_x_coordinate(curr)), ys.scale(instance.get_y_coordinate(curr)));
                glVertex2f(xs.scale(instance.get_x_coordinate(next)), ys.scale(instance.get_y_coordinate(next)));

                curr = next;


            } while (true);


            glEnd();

            /*glPushAttrib(GL_ENABLE_BIT);
            glLineStipple(1, 0xAAAA);
            glEnable(GL_LINE_STIPPLE);
            glColor4f(load_ratio,1-load_ratio,0.0, 0.1);
            glBegin(GL_LINES);
            glVertex2d(instance.get_x_coordinate(solution.get_last_customer(route)),
            instance.get_y_coordinate(solution.get_last_customer(route))); glVertex2d(instance.get_x_coordinate(instance.get_depot()),
            instance.get_y_coordinate(instance.get_depot())); glEnd(); glPopAttrib();*/
        }

        glColor4f(1.0, 0.75, 0.00, 1.0);
        glPointSize(20);
        glBegin(GL_POINTS);
        glVertex2d(xs.scale(instance.get_x_coordinate(instance.get_depot())), ys.scale(instance.get_y_coordinate(instance.get_depot())));
        glEnd();
        glPointSize(1);
    }

private:
    GLFWwindow* window = nullptr;

    const cobra::Instance& instance;
    double initial_cost;

    struct TrajectoryPoint {
        double shaken_solution_gap;
        double local_optima_gap;
        double current_solution_gap;
        double best_solution_gap;
    };

    double min_gap = std::numeric_limits<double>::max();
    double max_gap = std::numeric_limits<double>::min();

    std::vector<TrajectoryPoint> trajectory;

    double x_min{}, x_max{}, y_min{}, y_max{};

    class Scaler {
    public:
        Scaler() = default;

        void set(double in_min_, double in_max_, double out_min_, double out_max_) {
            in_min = in_min_;
            in_max = in_max_;
            out_min = out_min_;
            out_max = out_max_;
            scaled_min = scale(in_min);
            scaled_max = scale(in_max);
        }

        Scaler(double in_min_, double in_max_, double out_min_, double out_max_) {
            set(in_min_, in_max_, out_min_, out_max_);
        }

        double scale(double v) const {
            return (v - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
        }

    private:
        double in_min = 0, in_max = 10, out_min = 0, out_max = 10, scaled_min = 0, scaled_max = 10;
    };
};

#endif