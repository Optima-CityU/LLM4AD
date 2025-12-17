#ifndef _FILO2_LRUCACHE_HPP_
#define _FILO2_LRUCACHE_HPP_

#include <cassert>
#include <vector>


namespace cobra {

    class LRUCache {

    public:
        LRUCache(int capacity_, int vertices_num_) : cache_size(capacity_), cache(vertices_num_), counter(0) { }

        LRUCache(const LRUCache& other)
            : cache_size(other.cache_size), cache(other.cache), counter(other.counter), head(other.head), tail(other.tail) { }

        LRUCache& operator=(const LRUCache& other) {
            cache_size = other.cache_size;
            cache = other.cache;
            counter = other.counter;
            head = other.head;
            tail = other.tail;
            return *this;
        }

        void insert(int vertex) {

            if (cache[vertex].used) {

                remove(vertex);
                splay_on_top(vertex);

            } else {

                // Reached the cache size, evict the least recently used entry.
                if (counter == cache_size) {
                    remove(tail);
                } else {
                    counter++;
                }
                // Move the recently accessed entry to top.
                splay_on_top(vertex);
            }
        }

        void clear() {
            counter = 0;

            auto curr = head;
            while (curr != Entry::dummy_vertex) {
                const auto next = cache[curr].next;
                cache[curr].used = false;
                cache[curr].next = Entry::dummy_vertex;
                cache[curr].prev = Entry::dummy_vertex;
                curr = next;
            }

            head = Entry::dummy_vertex;
            tail = Entry::dummy_vertex;
        }

        int size() const {
            return counter;
        }

        bool empty() const {
            return counter == 0;
        }

        inline int begin() const {
            return head;
        }
        inline int last() const {
            return tail;
        }
        inline int get_next(int vertex) const {
            return cache[vertex].next;
        }
        inline int get_prev(int vertex) const {
            return cache[vertex].prev;
        }

        inline int end() const {
            return Entry::dummy_vertex;
        }

    private:
        class Entry {
        public:
            int prev = dummy_vertex;
            int next = dummy_vertex;
            bool used = false;
            static constexpr int dummy_vertex = -1;
        };

        int cache_size;
        std::vector<Entry> cache;
        int counter = 0;
        int head = Entry::dummy_vertex;
        int tail = Entry::dummy_vertex;

        inline void remove(int vertex) {

            assert(vertex != Entry::dummy_vertex);
            assert(cache[vertex].used);

            const auto prevEntry = cache[vertex].prev;
            const auto nextEntry = cache[vertex].next;

            // Head remove.
            if (prevEntry == Entry::dummy_vertex) {
                head = nextEntry;
            } else {
                cache[prevEntry].next = nextEntry;
            }

            // Tail remove.
            if (nextEntry == Entry::dummy_vertex) {
                tail = prevEntry;
            } else {
                cache[nextEntry].prev = prevEntry;
            }

            cache[vertex].used = false;
            cache[vertex].prev = Entry::dummy_vertex;
            cache[vertex].next = Entry::dummy_vertex;
        }

        inline void splay_on_top(int vertex) {

            assert(vertex != Entry::dummy_vertex);
            assert(!cache[vertex].used);

            cache[vertex].used = true;

            cache[vertex].next = head;
            if (head != Entry::dummy_vertex) {
                cache[head].prev = vertex;
            }
            head = vertex;
            cache[vertex].prev = Entry::dummy_vertex;

            if (tail == Entry::dummy_vertex) {
                tail = head;
            }
        }
    };

}  // namespace cobra

#endif