#pragma once

#include <cstdint>
#include <cstdio>

#include "hle/common.h"

namespace bb::libc {

inline FILE* open_file(const char* path, const char* mode) {
    FILE* f = std::fopen(path, mode);
    if (!f) set_guest_errno_host(errno);
    return f;
}
inline FILE* reopen_file(const char* path, const char* mode, FILE* file) {
    FILE* f = std::freopen(path, mode, file);
    if (!f) set_guest_errno_host(errno);
    return f;
}
template <typename Fn> inline auto file_result(FILE* file, Fn&& call) {
    errno = 0;
    auto result = call();
    const int error = errno;
    // EOF/short transfers are not necessarily errors; don't leak old host
    // errno or erase the guest's previous errno on successful operations.
    if (error && std::ferror(file)) set_guest_errno_host(error);
    return result;
}

struct Fpos {
    std::int64_t offset;
    std::uint64_t state[2];
};
static_assert(sizeof(Fpos) == 24);

inline int seek(FILE* file, std::int64_t offset, int whence) {
#if defined(_WIN32)
    return _fseeki64(file, offset, whence);
#else
    static_assert(sizeof(off_t) == 8);
    return ::fseeko(file, static_cast<off_t>(offset), whence);
#endif
}
inline std::int64_t tell(FILE* file) {
#if defined(_WIN32)
    return _ftelli64(file);
#else
    return static_cast<std::int64_t>(::ftello(file));
#endif
}
inline int get_position(FILE* file, Fpos* position) {
    const auto offset = tell(file);
    if (offset == -1) return -1;
    // Guest file I/O currently implements narrow streams only; no pending
    // multibyte conversion state. Never expose the host's opaque fpos_t.
    *position = {offset, {0, 0}};
    return 0;
}

}  // namespace bb::libc
