#ifndef _FILO2_TIMER_HPP_
#define _FILO2_TIMER_HPP_

#include <chrono>

namespace cobra {
    class Timer {
    public:
        Timer() {
            reset();
        }

        template <typename Units = typename std::chrono::high_resolution_clock::duration>
        unsigned long elapsed_time() const {
            auto counted_time = std::chrono::duration_cast<Units>(std::chrono::high_resolution_clock::now() - start_point).count();
            return static_cast<unsigned long>(counted_time);
        }

        void reset() {
            start_point = std::chrono::high_resolution_clock::now();
        }

    private:
        std::chrono::high_resolution_clock::time_point start_point;
    };
}  // namespace cobra

#endif