#ifndef _FILO2_NONCOPYABLE_HPP_
#define _FILO2_NONCOPYABLE_HPP_

namespace cobra {

    // Non-copyable mixin inspired by https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Non-copyable_Mixin.
    template <class T>
    class NonCopyable {
    public:
        NonCopyable(const NonCopyable&) = delete;
        NonCopyable(NonCopyable&&) noexcept = default;
        NonCopyable& operator=(const NonCopyable&) = delete;
        NonCopyable& operator=(NonCopyable&&) noexcept = delete;
        T& operator=(const T&) = delete;
        T& operator=(T&&) noexcept = delete;

    protected:
        NonCopyable() = default;
        ~NonCopyable() = default;  /// Protected non-virtual destructor
    };

}  // namespace cobra

#endif