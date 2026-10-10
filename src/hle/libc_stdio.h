#pragma once

#include <cstdint>
#include <cstdio>

namespace bb::libc {

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
