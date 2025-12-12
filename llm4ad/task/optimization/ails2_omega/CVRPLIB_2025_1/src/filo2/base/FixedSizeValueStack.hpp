#ifndef _FILO2_FIXEDSIZEVALUESTACK_HPP_
#define _FILO2_FIXEDSIZEVALUESTACK_HPP_

#include <cassert>
#include <functional>

namespace cobra {

    template <class T>
    class FixedSizeValueStack {
    public:
        explicit FixedSizeValueStack(int dimension, std::function<T(int index)> array_initializer) {
            assert(dimension > 0);
            array.resize(dimension);
            capacity = dimension;
            begin = 0;
            initializer = array_initializer;
            reset();
        }

        ~FixedSizeValueStack() = default;

        FixedSizeValueStack<T>& operator=(const FixedSizeValueStack<T>& other) {
            assert(capacity == other.capacity);

            for (int i = 0; i < capacity; i++) {
                array[i] = other.array[i];
            }
            begin = other.begin;
            initializer = other.initializer;
            return *this;
        }

        T get() {
            assert(begin < capacity);
            auto item = array[begin];
            begin++;
            return item;
        }

        void push(T item) {
            begin--;
            assert(begin >= 0);
            array[begin] = item;
        }

        void reset() {
            for (int i = 0; i < capacity; i++) {
                array[i] = initializer(i);
            }
            begin = 0;
        }

        int size() const {
            return capacity - begin;
        }

        bool is_empty() {
            return begin == capacity;
        }

    private:
        std::vector<T> array;
        int begin = 0;
        int capacity;
        std::function<T(int)> initializer = nullptr;
    };

}  // namespace cobra

#endif