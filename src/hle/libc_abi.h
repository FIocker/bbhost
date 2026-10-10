#pragma once

#include <cstddef>
#include <cstdint>

namespace bb::libc {

// Dinkumware's layouts, verified against sce_module/libc.prx. Orbis uses
// LP64 integers, but its libc tm/lconv are NOT the FreeBSD/glibc structures.
struct Div64 {
    std::int64_t quot, rem;
};
static_assert(sizeof(Div64) == 16 && offsetof(Div64, rem) == 8);
inline Div64 divide(std::int64_t n, std::int64_t d) { return {n / d, n % d}; }

}  // namespace bb::libc
