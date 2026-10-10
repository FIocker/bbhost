#pragma once

#include <cstddef>
#include <cstdint>
#include <ctime>

namespace bb::libc {

// Dinkumware's layouts, verified against sce_module/libc.prx. Orbis uses
// LP64 integers, but its libc tm/lconv are NOT the FreeBSD/glibc structures.
struct Div64 {
    std::int64_t quot, rem;
};
static_assert(sizeof(Div64) == 16 && offsetof(Div64, rem) == 8);
inline Div64 divide(std::int64_t n, std::int64_t d) { return {n / d, n % d}; }

struct Tm {
    std::int32_t sec, min, hour, mday, mon, year, wday, yday, isdst;
};
static_assert(sizeof(Tm) == 36 && offsetof(Tm, isdst) == 32);
inline void tm_from_host(const std::tm& s, Tm* d) {
    *d = {s.tm_sec, s.tm_min, s.tm_hour, s.tm_mday, s.tm_mon,
          s.tm_year, s.tm_wday, s.tm_yday, s.tm_isdst};
}
inline std::tm tm_to_host(const Tm* g) {
    std::tm t{};
    t.tm_sec = g->sec;
    t.tm_min = g->min;
    t.tm_hour = g->hour;
    t.tm_mday = g->mday;
    t.tm_mon = g->mon;
    t.tm_year = g->year;
    t.tm_wday = g->wday;
    t.tm_yday = g->yday;
    t.tm_isdst = g->isdst;
    return t;
}

}  // namespace bb::libc
