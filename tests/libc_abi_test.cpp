#include "hle/libc_abi.h"

#include <cstdio>
#include <cstdlib>
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
    std::puts("libc ABI tests passed");
}
