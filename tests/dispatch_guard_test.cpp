// host/dispatch_guard.h: the indirect-dispatch limit host_gpu_dispatch keeps
// to, and the command processor's record of small copies that names where a
// dispatch's arguments came from.
#include "host/dispatch_guard.h"

#include <cstdint>
#include <cstdio>

static int failures = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            ++failures;                                                  \
        }                                                                \
    } while (0)

int main() {
    using dispatch_guard::groups_fit;
    using dispatch_guard::SmallCopies;
    {
        // A particle system with no particles: GX's copy of a counter of 0.
        const std::uint32_t g[3] = {0, 1, 1};
        CHECK(groups_fit(g));
    }
    {
        const std::uint32_t g[3] = {65535, 65535, 65535};
        CHECK(groups_fit(g));
    }
    {
        const std::uint32_t x[3] = {65536, 1, 1}, y[3] = {1, 65536, 1}, z[3] = {1, 1, 65536};
        CHECK(!groups_fit(x));
        CHECK(!groups_fit(y));
        CHECK(!groups_fit(z));
    }
    {
        // The Forbidden Woods' (runs/pace-F1): a counter's float bits, 0xbd4ede96.
        const std::uint32_t g[3] = {3176062614u, 1, 1};
        CHECK(!groups_fit(g));
    }

    SmallCopies c;
    std::uint64_t src = 0;
    CHECK(!c.source_of(0x217da6cf0, src));
    c.note(0x217da6cf0, 0x22acb5d08, 4);  // GX's copy of a counter into a dispatch's x
    CHECK(c.source_of(0x217da6cf0, src) && src == 0x22acb5d08);
    CHECK(!c.source_of(0x217da6cf4, src));  // y was not copied
    CHECK(!c.source_of(0x217da6cec, src));
    c.note(0x1000, 0x2000, 12);
    CHECK(c.source_of(0x1008, src) && src == 0x2008);
    c.note(0x217da6cf0, 0x239a8b408, 4);  // a newer copy over the same word wins
    CHECK(c.source_of(0x217da6cf0, src) && src == 0x239a8b408);
    c.note(0x5000, 0x6000, 64);  // not a small copy: not kept
    CHECK(!c.source_of(0x5000, src));
    c.note(0x7000, 0x8000, 0);
    CHECK(!c.source_of(0x7000, src));
    for (std::uint64_t i = 0; i < 100; ++i) c.note(0x9000 + i * 16, 0xa000 + i * 16, 4);  // the oldest are forgotten
    CHECK(!c.source_of(0x217da6cf0, src));
    CHECK(!c.source_of(0x1008, src));
    CHECK(c.source_of(0x9000 + 99 * 16, src) && src == 0xa000 + 99 * 16);
    CHECK(c.source_of(0x9000 + 68 * 16, src) && src == 0xa000 + 68 * 16);
    CHECK(!c.source_of(0x9000 + 67 * 16, src));

    if (failures) return 1;
    std::printf("dispatch_guard_test: ok\n");
    return 0;
}
