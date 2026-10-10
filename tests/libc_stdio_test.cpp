#include "hle/libc_stdio.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); } } while (0)

int main() {
    using namespace bb::libc;
    FILE* f = std::tmpfile();
    CHECK(f);
    // Move the cursor only: no multi-gigabyte data is allocated or written.
    constexpr std::int64_t large = 0x100000123LL;
    CHECK(seek(f, large, SEEK_SET) == 0);
    CHECK(tell(f) == large);
    CHECK(seek(f, -17, SEEK_CUR) == 0);
    CHECK(tell(f) == large - 17);
    struct Guarded { Fpos pos; unsigned char guard[8]; } out;
    std::memset(&out, 0xa5, sizeof(out));
    CHECK(get_position(f, &out.pos) == 0);
    CHECK(out.pos.offset == large - 17 && out.pos.state[0] == 0 && out.pos.state[1] == 0);
    for (auto b : out.guard) CHECK(b == 0xa5);
    CHECK(seek(f, 0, SEEK_END) == 0 && tell(f) == 0);
    errno = 0;
    CHECK(seek(f, -1, SEEK_SET) == -1 && errno == EINVAL);
    CHECK(std::fclose(f) == 0);
    std::puts("libc stdio tests passed");
}
