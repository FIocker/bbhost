#include "hle/libc_abi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); } } while (0)

// Explicit SysV ABI also on Windows: guest ldiv returns quotient in RAX and
// remainder in RDX, not a packed pair of 32-bit Windows longs.
__attribute__((sysv_abi, noinline)) bb::libc::Div64 guest_div(std::int64_t n, std::int64_t d) {
    return bb::libc::divide(n, d);
}

int main() {
    using namespace bb::libc;
    const std::int64_t numerators[] = {0, 0xdeadbeefLL, 0x123456789abcdefLL,
                                       -0x123456789abcdefLL, INT64_MAX, INT64_MIN};
    for (const auto n : numerators) for (const auto d : {127773LL, -127773LL}) {
        const auto v = guest_div(n, d);
        CHECK(v.quot == n / d && v.rem == n % d);
        CHECK(v.quot * d + v.rem == n);
    }
    // Match the eboot's 36-byte localtime_s destination followed by a cookie.
    struct GuardedTm { Tm value; unsigned char tail[20]; } g{};
    std::memset(g.tail, 0xa5, sizeof(g.tail));
    std::tm host{};
    host.tm_sec = 37; host.tm_min = 24; host.tm_hour = 19;
    host.tm_mday = 11; host.tm_mon = 6; host.tm_year = 124;
    host.tm_wday = 4; host.tm_yday = 192; host.tm_isdst = 1;
    tm_from_host(host, &g.value);
    for (auto byte : g.tail) CHECK(byte == 0xa5);
    const auto roundtrip = tm_to_host(&g.value);
    Tm again{};
    tm_from_host(roundtrip, &again);
    CHECK(std::memcmp(&again, &g.value, sizeof(Tm)) == 0);
    char date[32];
    CHECK(std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M:%S", &roundtrip) == 19);
    CHECK(std::strcmp(date, "2024-07-11 19:24:37") == 0);

    Lconv lc;
    const char* decimal = nullptr;
    std::memcpy(&decimal, reinterpret_cast<const char*>(&lc) + 0x48, sizeof(decimal));
    CHECK(std::strcmp(decimal, ".") == 0);
    CHECK(*lc.currency_symbol == 0 && *lc.thousands_sep == 0);
    for (auto flag : lc.monetary_flags) CHECK(flag == 127);
    CHECK(std::strcmp(set_c_locale(nullptr), "C") == 0);
    CHECK(std::strcmp(set_c_locale(""), "C") == 0);
    CHECK(std::strcmp(set_c_locale("C"), "C") == 0);
    CHECK(!set_c_locale("POSIX") && !set_c_locale("en_US.UTF-8"));

    std::puts("libc ABI tests passed");
}
