#ifndef _FILO2_SMALLFLATMAP_HPP_
#define _FILO2_SMALLFLATMAP_HPP_


#include <cstddef>
#include <utility>
#include "functor.hpp"

namespace cobra {

    template <typename Key, typename Value, Key emptyKey, int maxSize, class op = identity_functor<Key>>
    class SmallFlatMap {

        using KVpair = std::pair<Key, Value>;
        using KVpair_ptr = KVpair*;

        static_assert(maxSize > 0, "Needs a positive value size.");
        static_assert(std::is_integral_v<decltype(maxSize)>, "Needs a integral type for maxSize");
        static_assert(maxSize * 5 / 4 <= (1 << 16), "Choose a smaller maxSize.");
        static constexpr int next2pow(int v) {
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            return ++v;
        }
        static_assert(next2pow(maxSize * 5 / 4) * sizeof(KVpair) <= (1 << 16), "Maximum memory occupation of 65KB.");

    public:
        class custom_iterator {
            friend class SmallFlatMap<Key, Value, emptyKey, maxSize, op>;

        private:
            custom_iterator(KVpair_ptr _base, KVpair_ptr _end) : base(_base), end(_end) {
                while (base != end && base->first == emptyKey) ++base;
            };

        public:
            inline auto& operator*() {
                return *base;
            }

            inline auto operator->() {
                return base;
            }

            inline auto& operator++() {
                do {
                    ++base;
                } while (base != end && base->first == emptyKey);
                return *this;
            }

            inline auto operator!=(const custom_iterator x) {
                return x.base != base;
            }

            inline auto operator==(const custom_iterator x) {
                return x.base == base;
            }

        private:
            KVpair_ptr base;
            const KVpair_ptr end;
        };

    public:
        SmallFlatMap() {
            for (auto& p : buffer) p.first = emptyKey;
        };

        inline std::pair<Key, Value>& find(const Key k) {
            auto index = op()(k) & realSizem1;
            auto key = buffer[index].first;
            while (key != k && key != emptyKey) {
                index = (index + 1) & realSizem1;
                key = buffer[index].first;
            }
            return buffer[index];
        }

        inline bool insert(const Key k, const Value v) {
            auto& candidate_place = find(k);
            if (candidate_place.first != emptyKey) return false;  // Element already there.

            candidate_place = {k, v};
            return true;
        }

        inline auto& operator[](Key k) {
            auto& kv = find(k);
            kv.first = k;  // No check on sz, we need to decide if we want to check or not.
            return kv.second;
        }

        inline void clear() {
            for (auto& p : buffer) p.first = emptyKey;
        }

        inline size_t count(Key k) {
            return static_cast<size_t>(find(k).first != emptyKey);
        }

        inline auto begin() {
            return custom_iterator(buffer, buffer + realSize);
        };

        inline auto end() {
            return custom_iterator(buffer + realSize, buffer + realSize);
        };

    private:
        // Why 5/4 do you ask? Clearly a well thought value, not at all the nearest
        // small fraction that transform 25 to ~32.
        constexpr static int realSize = next2pow(maxSize * 5 / 4);
        constexpr static int realSizem1 = realSize - 1;

    public:
        std::pair<Key, Value> buffer[realSize];
    };
}  // namespace cobra

#endif