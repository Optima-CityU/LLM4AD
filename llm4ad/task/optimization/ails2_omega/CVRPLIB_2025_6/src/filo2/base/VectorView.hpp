#ifndef _FILO2_VECTORVIEW_HPP_
#define _FILO2_VECTORVIEW_HPP_



#include <iterator>
namespace cobra {

    // Apply a function to a random access range online, without modifying the underlying values.
    template <typename IterT, template <typename T> class OpTmpl>
    class VectorView {
        using UnaryOp = OpTmpl<typename std::iterator_traits<IterT>::value_type>;

        static_assert(std::is_same_v<typename std::iterator_traits<IterT>::iterator_category, std::random_access_iterator_tag>,
                      "The iterator type need to be a random_access_iterator or "
                      "a pointer.");

    public:
        class custom_const_iterator {
        public:
            custom_const_iterator(IterT _original) : original(_original){};

            inline auto operator*() {
                return UnaryOp()(*original);
            }

            inline auto operator++() {
                ++original;
                return *this;
            }

            inline auto operator->() {
                return original;
            }

            inline auto operator-(int x) const {
                return custom_const_iterator(original - x);
            }

            inline auto operator-(custom_const_iterator x) const {
                return original - x.original;
            }

            inline auto operator!=(custom_const_iterator x) {
                return x.original != original;
            }

            inline auto operator==(custom_const_iterator x) {
                return x.original == original;
            }

        private:
            IterT original;
        };

        VectorView(IterT _first, IterT _last) : first(_first), last(_last){};

        inline auto operator[](size_t index) const {
            return UnaryOp()(first[index]);
        }

        inline auto at(size_t index) const {
            return UnaryOp()(first[index]);
        }

        inline auto begin() const {
            return custom_const_iterator(first);
        }

        inline auto end() const {
            return custom_const_iterator(last);
        }

        inline auto size() const {
            return std::distance(first, last);
        }

    private:
        IterT first;
        IterT last;
    };
}  // namespace cobra

#endif