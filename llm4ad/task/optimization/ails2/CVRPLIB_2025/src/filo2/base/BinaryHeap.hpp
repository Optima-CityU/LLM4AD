#ifndef _FILO2_BINARYHEAP_HPP_
#define _FILO2_BINARYHEAP_HPP_

#include <cassert>
#include <iostream>
#include <vector>

#include "functor.hpp"

#define LEFT(X) (2 * (X) + 1)
#define RIGHT(X) (2 * (X) + 2)
#define PARENT(X) (((X)-1) / 2)

namespace cobra {

    // Generic binary heap.
    template <typename T, class Cmp, class GetIdx, class SetIdx, class Updt, int unheaped = -1>
    class BinaryHeap {
    public:
        BinaryHeap() { }
        BinaryHeap(const BinaryHeap& bh) : heap(bh.heap) { }
        BinaryHeap(BinaryHeap&& bh) : heap(std::move(bh.heap)) { }

        // Resets the data structure.
        void reset() {
            for (auto& t : heap) {
                SetIdx()(t, unheaped);
            }
            heap.clear();
        }

        // Returns whether the heap is empty.
        bool empty() const {
            return heap.empty();
        }

        // Inserts an element.
        void insert(T elem) {
            const int hindex = heap.size();
            SetIdx()(elem, hindex);
            heap.emplace_back(elem);
            upsift(hindex);

            assert(is_heap());
        }

        // Returns the heap head.
        T get() {
            assert(!heap.empty());

            SetIdx()(heap[0], unheaped);
            auto elem = std::move(heap[0]);

            if (heap.size() == 1) {
                heap.pop_back();
                return elem;
            }

            SetIdx()(heap.back(), 0);
            heap[0] = std::move(heap.back());
            heap.pop_back();
            heapify(0);

            assert(is_heap());
            return elem;
        }

        // Removes a specific heaped entry given its heap index.
        void remove(int hindex) {

            if (hindex < static_cast<int>(heap.size()) - 1) {
                auto last = std::move(heap.back());
                heap.pop_back();
                replace(hindex, std::move(last));
            } else {
                SetIdx()(heap[hindex], unheaped);
                heap.pop_back();
            }

            assert(is_heap());
        }

        // Replaces the element at `hindex` with the given `elem`.
        void replace(int hindex, T elem) {
            assert(hindex >= 0 && hindex < static_cast<int>(heap.size()));

            const auto case3 = Cmp()(heap[hindex], elem);

            SetIdx()(heap[hindex], unheaped);
            SetIdx()(elem, hindex);
            heap[hindex] = elem;

            if (case3 > 0) {
                upsift(hindex);
            } else if (case3 < 0) {
                heapify(hindex);
            }

            assert(is_heap());
        }

        // Returns the number of heap elements.
        auto size() const {
            return heap.size();
        }

        // Returns the element at the given `hindex` without modifying the heap.
        T& spy(int hindex) {
            return heap[hindex];
        }

        template <typename... Args>
        void update(int hindex, Args&&... args) {
            assert(hindex >= 0 && hindex < static_cast<int>(heap.size()));

            const auto case3 = Updt()(heap[hindex], std::forward<Args>(args)...);

            if (case3 > 0) {
                upsift(hindex);
            } else if (case3 < 0) {
                heapify(hindex);
            }

            assert(is_heap());
        }

        // Implement to print out some info about the heap.
        virtual void dump() { }

    private:
        int inline min_lr(T& parent, int lindex, int rindex) {
            const int hsize = heap.size();
            auto smallest = lindex;
            if (rindex < hsize && Cmp()(heap[rindex], heap[lindex]) < 0) {
                smallest = rindex;
            }  // !! rindex < lindex always !!
            if (smallest < hsize && Cmp()(heap[smallest], parent) < 0) {
                return smallest;
            }
            return unheaped;
        }

        void heapify(int hindex) {

            auto smallest = min_lr(heap[hindex], LEFT(hindex), RIGHT(hindex));
            if (smallest == unheaped) {
                return;
            }

            auto elem = std::move(heap[hindex]);
            while (smallest != unheaped) {

                SetIdx()(heap[smallest], hindex);
                heap[hindex] = std::move(heap[smallest]);
                hindex = smallest;
                smallest = min_lr(elem, LEFT(hindex), RIGHT(hindex));
            }

            SetIdx()(elem, hindex);
            heap[hindex] = std::move(elem);
        }

        void upsift(int hindex) {

            if (hindex == 0) {
                return;
            }

            auto pindex = hindex;
            auto elem = std::move(heap[hindex]);
            while (hindex && Cmp()(elem, heap[pindex = PARENT(pindex)]) < 0) {

                SetIdx()(heap[pindex], hindex);
                heap[hindex] = std::move(heap[pindex]);
                hindex = pindex;
            }

            SetIdx()(elem, hindex);
            heap[hindex] = std::move(elem);
        }

        bool is_heap() {

            const int hsize = heap.size();
            for (int n = 0; n < hsize; ++n) {
                const auto& t = heap[n];
                if (auto idx = GetIdx()(t); idx != n) {
                    std::cout << "Heap index of element n is not n\n";
                    std::cout << idx << " " << n << "\n";
                    dump();
                    return false;
                }
            }

            for (int n = 0; n < hsize; ++n) {
                const auto lindex = LEFT(n);
                const auto rindex = RIGHT(n);
                if (lindex < hsize) {
                    if (Cmp()(heap[lindex], heap[n]) < 0) {
                        std::cout << "left: " << n << " > " << lindex << "\n";
                        dump();
                        return false;
                    }
                }
                if (rindex < hsize) {
                    if (Cmp()(heap[rindex], heap[n]) < 0) {
                        std::cout << "right: " << n << " > " << rindex << "\n";
                        dump();
                        return false;
                    }
                }
            }

            return true;
        }

        std::vector<T> heap;
    };

    template <typename N, auto field>
    struct CmpFieldPtr {
        auto operator()(N* r1, N* r2) {
            assert(r1 && r2);
            return access_field_functor<N, field>()(*r1) - access_field_functor<N, field>()(*r2);
        }
    };

    template <typename N, auto field>
    struct GetIdxFieldPtr {
        auto operator()(N* r1) {
            assert(r1);
            return access_field_functor<N, field>()(*r1);
        }
    };

    template <typename N, auto field>
    struct SetIdxFieldPtr {
        void operator()(N* r1, int idx) {
            assert(r1);
            access_field_functor<N, field>()(*r1) = idx;
        }
    };

    template <typename N, auto field>
    struct UpdtFieldPtr {
        auto operator()(N* r1, typename std::decay<decltype(std::declval<N>().*field)>::type val) {
            assert(r1);
            const auto res = access_field_functor<N, field>()(*r1) - val;
            access_field_functor<N, field>()(*r1) = val;
            return res;
        }
    };

    template <typename N, auto findex, auto fvalue>
    class BinaryHeapPtr
        : public BinaryHeap<N*, CmpFieldPtr<N, fvalue>, GetIdxFieldPtr<N, findex>, SetIdxFieldPtr<N, findex>, UpdtFieldPtr<N, fvalue>> { };

}  // namespace cobra

#endif