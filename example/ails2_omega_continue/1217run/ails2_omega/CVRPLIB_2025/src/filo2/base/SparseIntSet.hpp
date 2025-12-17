#ifndef _FILO2_SPARSEINTSET_HPP_
#define _FILO2_SPARSEINTSET_HPP_

#include <cassert>
#include <cmath>
#include <vector>

namespace cobra {

    class SparseIntSet {

    public:
        explicit SparseIntSet(unsigned int entries_num) {
            flags.resize(entries_num);
        }

        SparseIntSet(const SparseIntSet& other) {
            flags = other.flags;
        }

        SparseIntSet& operator=(const SparseIntSet& other) {
            flags = other.flags;
            return *this;
        }

        inline void insert(int value) {
            const auto already_here = contains(value);
            if (!already_here) {
                insert_without_checking_existance(value);
            }
        }

        inline void insert_without_checking_existance(int value) {
            assert(value < static_cast<int>(flags.size()));
            flags[value] = true;
            elements.push_back(value);
        }

        inline bool contains(int value) const {
            assert(value < static_cast<int>(flags.size()));
            return flags[value];
        }

        void clear() {
            for (auto value : elements) {
                assert(value < static_cast<int>(flags.size()));
                flags[value] = false;
            }
            elements.clear();
        }

        const std::vector<int>& get_elements() const {
            return elements;
        }

        unsigned int size() const {
            return elements.size();
        }

    private:
        std::vector<bool> flags;
        std::vector<int> elements;
    };

}  // namespace cobra

#endif