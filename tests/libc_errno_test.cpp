#include "hle/common.h"

#include <cstdio>

thread_local int t_guest_errno;
#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); } } while (0)

int main() {
    CHECK(to_freebsd(0) == 0 && to_freebsd(ENOENT) == 2 && to_freebsd(EINVAL) == 22);
    CHECK(to_freebsd(EAGAIN) == 35 && to_freebsd(EDEADLK) == 11);
    CHECK(to_freebsd(ENAMETOOLONG) == 63 && to_freebsd(ENOTEMPTY) == 66);
    CHECK(to_freebsd(ENOLCK) == 77 && to_freebsd(ENOSYS) == 78 && to_freebsd(EILSEQ) == 86);
    CHECK(to_freebsd(ECONNREFUSED) == 61 && to_freebsd(EADDRINUSE) == 48);
    CHECK(to_freebsd(EINPROGRESS) == 36 && to_freebsd(EOVERFLOW) == 84);
    set_guest_errno_host(EAGAIN);
    CHECK(t_guest_errno == 35);
    CHECK(libc_result(0) == 0 && t_guest_errno == 35);
    errno = ENOENT;
    CHECK(libc_result(-1) == -1 && t_guest_errno == 2);
    CHECK(static_cast<unsigned>(sce_err(EAGAIN)) == 0x80020023u);
    std::puts("libc errno tests passed");
}
