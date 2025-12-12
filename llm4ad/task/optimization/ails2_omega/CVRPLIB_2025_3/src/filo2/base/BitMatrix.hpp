#ifndef _FILO2_BITMATRIX_HPP_
#define _FILO2_BITMATRIX_HPP_

#include <cmath>
#include <vector>

#include "NonCopyable.hpp"
#include "SmallFlatSet.hpp"

namespace cobra {
    template <int maxSize>
    class BitMatrix : NonCopyable<BitMatrix<maxSize>> {
    public:
        BitMatrix(int rows) : data(rows) { }

        inline void reset(int row) {
            data[row].clear();
        }

        inline void set(int row, int entry) {
            data[row].insert(entry);
        }

        inline bool is_set(int row, int entry) {
            return static_cast<bool>(data[row].count(entry));
        }

        inline void overwrite(int source_row, int destination_row) {
            data[destination_row] = data[source_row];
        }

        inline auto& get_set_entries_possibly_with_duplicates(int row) {
            return data[row];
        }

    private:
        std::vector<SmallFlatSet<unsigned int, static_cast<unsigned int>(~0), maxSize>> data;
    };

}  // namespace cobra

#endif
