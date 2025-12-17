#ifndef _FILO2_WELFORD_HPP_
#define _FILO2_WELFORD_HPP_

namespace cobra {

    // https://gist.github.com/alexalemi/2151722
    class Welford {

    public:
        Welford() = default;

        Welford(const Welford& other) {
            k = other.k;
            mean = other.mean;
        }

        Welford& operator=(const Welford& other) = default;

        void update(double x) {
            ++k;
            const auto new_mean = mean + (x - mean) * 1.0 / static_cast<double>(k);
            mean = new_mean;
        }

        double get_mean() const {
            return mean;
        }

        void reset() {
            k = 0;
            mean = 0.0;
        }

    private:
        unsigned long k = 0;
        double mean = 0.0;
    };

}  // namespace cobra

#endif